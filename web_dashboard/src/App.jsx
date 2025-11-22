import React, { useEffect, useState, useRef } from "react";
import { createClient } from "@supabase/supabase-js";
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  Image as ImageIcon,
  Users,
  Send,
  Bot,
} from "lucide-react";
import { runRAGAnalysis } from "./ragAnalyzer";

// --- Supabase setup ---
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "";
const SUPABASE_KEY =
  import.meta.env.VITE_SUPABASE_KEY ||
  import.meta.env.VITE_SUPABASE_ANON_KEY ||
  "";

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.warn(
    "Missing Supabase env vars: VITE_SUPABASE_URL and VITE_SUPABASE_KEY (or VITE_SUPABASE_ANON_KEY)"
  );
}

let supabase = null;
if (SUPABASE_URL && SUPABASE_KEY) {
  supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
} else {
  supabase = null;
}

// --- Helper functions ---
async function fetchStudents() {
  if (!supabase || typeof supabase.from !== "function") return {};
  try {
    const { data, error } = await supabase
      .from("students")
      .select("roll_no,full_name");
    if (error) {
      console.error("fetchStudents error", error);
      return {};
    }
    const map = {};
    (data || []).forEach((s) => (map[s.roll_no] = s.full_name));
    return map;
  } catch (err) {
    console.error("fetchStudents exception", err);
    return {};
  }
}

async function fetchLastLogs(limit = 20) {
  if (!supabase || typeof supabase.from !== "function") return [];
  try {
    const { data, error } = await supabase
      .from("attendance_logs")
      .select("roll_no,event_type,timestamp,is_spoof,evidence_url")
      .order("timestamp", { ascending: false })
      .limit(limit);

    if (error) {
      console.error("fetchLastLogs error", error);
      return [];
    }
    return data || [];
  } catch (err) {
    console.error("fetchLastLogs exception", err);
    return [];
  }
}

async function calculateStudentAttendance(rollNo, allStudentsMap) {
  if (!supabase || !rollNo) return { duration: "N/A", percentage: "—" };

  try {
    const { data: logs, error } = await supabase
      .from("attendance_logs")
      .select("event_type, timestamp, is_spoof")
      .eq("roll_no", rollNo)
      .eq("is_spoof", false)
      .order("timestamp", { ascending: true });

    if (error) {
      console.error("Attendance calculation error:", error);
      return { duration: "DB Error", percentage: "—" };
    }

    if (!logs || logs.length === 0) {
      return { duration: "0h 0m", percentage: "0%" };
    }

    let totalDurationMinutes = 0;
    let entryTime = null;

    logs.forEach((log) => {
      const logTime = new Date(log.timestamp).getTime();
      if (log.event_type === "ENTRY") {
        if (entryTime === null) entryTime = logTime;
      } else if (log.event_type === "EXIT") {
        if (entryTime !== null && logTime > entryTime) {
          totalDurationMinutes += (logTime - entryTime) / 60000;
          entryTime = null;
        }
      }
    });

    const TOTAL_SCHEDULED_MINUTES = 480;
    const percentage = Math.min(
      100,
      (totalDurationMinutes / TOTAL_SCHEDULED_MINUTES) * 100
    ).toFixed(1);
    const hours = Math.floor(totalDurationMinutes / 60);
    const minutes = Math.round(totalDurationMinutes % 60);
    const durationString = `${hours}h ${minutes}m`;

    return { duration: durationString, percentage: `${percentage}%` };
  } catch (err) {
    console.error("Critical calculation failure", err);
    return { duration: "Fail", percentage: "—" };
  }
}

// --- MAIN APP COMPONENT ---

export default function App() {
  const [studentsMap, setStudentsMap] = useState({});
  const [events, setEvents] = useState([]);
  const [occupancy, setOccupancy] = useState(0);

  const [searchRoll, setSearchRoll] = useState("");
  const [studentInfo, setStudentInfo] = useState(null);

  const [gallery, setGallery] = useState([]);
  const [assistantQuery, setAssistantQuery] = useState("");
  const [assistantResponse, setAssistantResponse] = useState(
    "How can I help you analyze the security logs?"
  );
  const [isLoadingAI, setIsLoadingAI] = useState(false);

  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    (async () => {
      try {
        const map = await fetchStudents();
        const logs = await fetchLastLogs(20);
        if (!mounted.current) return;
        setStudentsMap(map);
        setEvents(logs);
        setGallery(
          (logs || [])
            .filter((l) => (l.is_spoof || !l.roll_no) && l.evidence_url)
            .slice(0, 8)
        );
        computeOccupancyFromLogs(logs || []);
      } catch (e) {
        console.error("Initial load error", e);
      }
    })();

    let channel = null;
    if (supabase && typeof supabase.channel === "function") {
      channel = supabase
        .channel("public:attendance_logs")
        .on(
          "postgres_changes",
          { event: "INSERT", schema: "public", table: "attendance_logs" },
          (payload) => {
            const newRow = payload.new;
            setEvents((prev) => [newRow].concat(prev).slice(0, 100));
            setOccupancy((o) => computeDelta(o, newRow));
            const hasRollNew =
              newRow.roll_no !== null &&
              newRow.roll_no !== undefined &&
              String(newRow.roll_no).trim() !== "";
            if ((newRow.is_spoof || !hasRollNew) && newRow.evidence_url) {
              setGallery((g) => [newRow].concat(g).slice(0, 8));
            }
          }
        )
        .subscribe();
    }

    return () => {
      mounted.current = false;
      if (channel && supabase) supabase.removeChannel(channel);
    };
  }, []);

  function computeDelta(currentOccupancy, row) {
    if (!row) return currentOccupancy;
    if (row.event_type === "ENTRY") return currentOccupancy + 1;
    if (row.event_type === "EXIT") return Math.max(0, currentOccupancy - 1);
    return currentOccupancy;
  }

  function computeOccupancyFromLogs(logs) {
    let e = 0;
    logs.forEach((r) => {
      if (r.event_type === "ENTRY") e += 1;
      if (r.event_type === "EXIT") e -= 1;
    });
    setOccupancy(Math.max(0, e));
  }

  function computeUniqueInAndIntruders(logs) {
    if (!logs || logs.length === 0) return { uniqueIn: 0, intruderCount: 0 };
    const rollLatest = new Map();
    const intruderSeen = new Set();

    for (const ev of logs) {
      const hasRoll =
        ev.roll_no !== null &&
        ev.roll_no !== undefined &&
        String(ev.roll_no).trim() !== "";
      if (hasRoll) {
        const r = String(ev.roll_no);
        if (!rollLatest.has(r)) rollLatest.set(r, ev.event_type);
      } else {
        const key = ev.evidence_url || ev.timestamp || Math.random();
        if (!intruderSeen.has(key) && ev.event_type === "ENTRY") {
          intruderSeen.add(key);
        }
      }
    }
    let uniqueIn = 0;
    for (const [, evt] of rollLatest.entries()) if (evt === "ENTRY") uniqueIn++;
    return { uniqueIn, intruderCount: intruderSeen.size };
  }

  async function onSearchStudent() {
    if (!searchRoll) return;
    const name = studentsMap[searchRoll];
    setStudentInfo({
      roll_no: searchRoll,
      full_name: name || "Unknown",
      attendance_pct: name ? "Calculating..." : "Student Not Enrolled",
    });
    if (!name) return;
    const result = await calculateStudentAttendance(searchRoll, studentsMap);
    setStudentInfo({
      roll_no: searchRoll,
      full_name: name,
      attendance_pct: result.percentage,
    });
  }

  async function onAssistantSend() {
    if (!assistantQuery || isLoadingAI || !supabase) return;
    setIsLoadingAI(true);
    setAssistantResponse("AI: Analyzing database...");
    try {
      const result = await runRAGAnalysis(supabase, assistantQuery);
      setAssistantResponse(result);
    } catch (error) {
      setAssistantResponse("Error processing AI request.");
    } finally {
      setIsLoadingAI(false);
    }
    setAssistantQuery("");
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-900 text-white font-sans overflow-hidden">
      <header className="px-6 py-4 border-b border-slate-800 bg-slate-900 z-10">
        <div className="max-w-7xl mx-auto flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
            <h1 className="text-xl font-bold tracking-wide text-slate-100">
              BIOMETRIC SENTRY <span className="text-green-500">LIVE</span>
            </h1>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-md border border-slate-700">
              <Users size={18} className="text-blue-400" />
              <div className="text-xs uppercase text-slate-400 tracking-wider">
                Occupancy
              </div>
              <div className="ml-2 text-xl font-mono font-bold text-blue-400">
                {computeUniqueInAndIntruders(events).uniqueIn}
              </div>
            </div>
            <div className="flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-md border border-slate-700">
              <AlertTriangle size={18} className="text-red-400" />
              <div className="text-xs uppercase text-slate-400 tracking-wider">
                Threats
              </div>
              <div className="ml-2 text-xl font-mono font-bold text-red-400">
                {computeUniqueInAndIntruders(events).intruderCount}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full p-6 overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          {/* LEFT COLUMN: LIVE FEED (Wide) */}
          <div className="lg:col-span-2 flex flex-col gap-4 h-full overflow-hidden">
            <div className="bg-slate-800 rounded-lg border border-slate-700 flex flex-col h-full shadow-lg">
              <div className="p-4 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Users className="text-green-400" size={20} /> Live Event Feed
                </h2>
                <span className="text-xs font-mono text-slate-500">
                  REALTIME_WS: CONNECTED
                </span>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin scrollbar-thumb-slate-600">
                {events.slice(0, 20).map((ev, i) => {
                  const hasRoll =
                    ev.roll_no && String(ev.roll_no).trim() !== "";
                  const isSpoof = !!ev.is_spoof;
                  const name = hasRoll
                    ? studentsMap[ev.roll_no] || ev.roll_no
                    : "UNKNOWN";

                  let borderColor = "border-l-slate-500";
                  let bgColor = "bg-slate-700/30";
                  let Icon = Users;
                  let iconColor = "text-slate-400";
                  let badge = null;

                  if (isSpoof) {
                    borderColor = "border-l-red-500";
                    bgColor = "bg-red-900/20";
                    Icon = XCircle;
                    iconColor = "text-red-400";
                    badge = (
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-red-900/50 text-red-200 border border-red-500/50">
                        PROXY
                      </span>
                    );
                  } else if (!hasRoll) {
                    borderColor = "border-l-amber-500";
                    bgColor = "bg-amber-900/20";
                    Icon = AlertTriangle;
                    iconColor = "text-amber-400";
                    badge = (
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-900/50 text-amber-200 border border-amber-500/50">
                        INTRUDER
                      </span>
                    );
                  } else if (ev.event_type === "ENTRY") {
                    borderColor = "border-l-green-500";
                    bgColor = "bg-green-900/10";
                    Icon = CheckCircle;
                    iconColor = "text-green-400";
                  }

                  return (
                    <div
                      key={ev.id || i}
                      className={`p-3 rounded-r-md border-l-4 ${borderColor} ${bgColor} flex items-start gap-4 transition-all hover:bg-slate-700/50`}
                    >
                      <div className="mt-1">
                        <Icon className={iconColor} size={20} />
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium text-slate-200 flex items-center gap-2">
                              {name} {badge}
                            </div>
                            <div className="text-xs text-slate-400 font-mono mt-0.5 uppercase tracking-wide">
                              {ev.event_type} • CAM-{ev.camera_id}
                            </div>
                          </div>
                          <div className="text-xs text-slate-500 font-mono">
                            {new Date(ev.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                        {/* Inline Evidence for Bad Actors */}
                        {(isSpoof || !hasRoll) && ev.evidence_url && (
                          <div className="mt-3">
                            <img
                              src={ev.evidence_url}
                              className="h-24 w-auto rounded border border-slate-600 object-cover"
                              alt="evidence"
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: ANALYTICS & CONTROLS (Narrow) */}
          <div className="flex flex-col gap-4 h-full overflow-y-auto scrollbar-hide">
            {/* 1. Student Lookup */}
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-md">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wider">
                Student Lookup
              </h3>
              <div className="flex gap-2 mb-4">
                <input
                  className="flex-1 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-green-500 outline-none"
                  placeholder="Enter Roll Number"
                  value={searchRoll}
                  onChange={(e) => setSearchRoll(e.target.value)}
                />
                <button
                  onClick={onSearchStudent}
                  className="bg-slate-700 hover:bg-slate-600 px-3 py-2 rounded border border-slate-600 text-slate-200 text-sm"
                >
                  Search
                </button>
              </div>
              {studentInfo && (
                <div className="bg-slate-700/50 rounded p-3 border border-slate-600">
                  <div className="text-xs text-slate-400 uppercase">
                    Student Profile
                  </div>
                  <div className="font-medium text-lg text-white mt-1">
                    {studentInfo.full_name}
                  </div>
                  <div className="text-xs text-slate-400 mt-2">
                    ID: {studentInfo.roll_no}
                  </div>
                  <div className="flex justify-between items-end mt-3 pt-3 border-t border-slate-600">
                    <div className="text-xs text-slate-400">Attendance</div>
                    <div className="text-xl font-bold text-green-400">
                      {studentInfo.attendance_pct}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* 2. AI Assistant (RAG) - MOVED HERE */}
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-md flex flex-col gap-3">
              <h3 className="text-sm font-semibold text-cyan-400 mb-1 uppercase tracking-wider flex items-center gap-2">
                <Bot size={16} /> AI Security Analyst
              </h3>
              <div className="bg-slate-900 rounded p-3 text-sm text-slate-300 min-h-[80px] max-h-[150px] overflow-y-auto border border-slate-700">
                {isLoadingAI ? (
                  <span className="animate-pulse text-cyan-500">
                    Analyzing logs...
                  </span>
                ) : (
                  assistantResponse
                )}
              </div>
              <div className="flex gap-2">
                <input
                  className="flex-1 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-xs text-white focus:border-cyan-500 outline-none"
                  placeholder="Ask about proxies..."
                  value={assistantQuery}
                  onChange={(e) => setAssistantQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onAssistantSend()}
                />
                <button
                  onClick={onAssistantSend}
                  disabled={isLoadingAI}
                  className="bg-cyan-900/30 hover:bg-cyan-900/50 border border-cyan-700 text-cyan-400 px-3 rounded"
                >
                  <Send size={14} />
                </button>
              </div>
            </div>

            {/* 3. Evidence Vault */}
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-md flex-1">
              <h3 className="text-sm font-semibold text-red-400 mb-3 uppercase tracking-wider flex justify-between items-center">
                Evidence Vault
                <span className="text-[10px] bg-red-900/30 text-red-300 px-1.5 py-0.5 rounded">
                  LAST 6
                </span>
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {gallery.length === 0 && (
                  <div className="col-span-2 text-xs text-slate-500 text-center py-4">
                    No threats detected yet.
                  </div>
                )}
                {gallery.slice(0, 6).map((g, i) => (
                  <div
                    key={i}
                    className="relative group rounded overflow-hidden border border-slate-600 bg-black aspect-video"
                  >
                    {g.evidence_url ? (
                      <img
                        src={g.evidence_url}
                        className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                        alt="threat"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-600">
                        <ImageIcon size={16} />
                      </div>
                    )}
                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 p-1">
                      <div className="text-[10px] font-bold text-white text-center">
                        {g.is_spoof ? "PROXY" : "INTRUDER"}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
