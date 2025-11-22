import { generateContent } from "./geminiApi";

// Ensure the Supabase client is initialized in your main App.jsx and passed here
// Or, if using a separate client file (recommended), import it here.
// For simplicity, we assume 'supabase' client is accessible or passed as an argument.
// In a real React app, you'd import the client initialized with your VITE keys.
// For this standalone function, we assume a supabase client object is available.
// If you are using a dedicated client file (e.g., supabaseClient.js), replace this line:
// import { supabase } from './supabaseClient';

const ANALYST_PROMPT =
  "You are a specialized security and attendance analyst named 'The Sentry AI'. Your task is to analyze the provided raw attendance data (which is in JSON format) and answer the user's question clearly and conversationally. Do not include the raw JSON data in your final answer. State the total number of threats found.";

/**
 * 1. Retrieves specific data from Supabase based on the user's question keywords.
 * 2. Augments the query with the raw data.
 * 3. Sends the query to the Gemini API for analysis.
 * @param {object} supabaseClient - The initialized Supabase client object.
 * @param {string} userQuestion - The question asked by the user (e.g., "how many proxies today").
 * @returns {Promise<string>} The AI's conversational response.
 */
export async function runRAGAnalysis(supabaseClient, userQuestion) {
  if (!supabaseClient) {
    return "Error: Database connection is offline. Cannot perform RAG analysis.";
  }

  const today = new Date().toISOString().split("T")[0];
  let queryBuilder = supabaseClient.from("attendance_logs").select("*");
  let dataLabel = "Attendance Logs";
  let filterApplied = false;

  // --- STEP 1: RETRIEVE DATA (Query Translation) ---

  // A. Check for PROXIES
  if (
    userQuestion.toLowerCase().includes("proxy") ||
    userQuestion.toLowerCase().includes("proxies")
  ) {
    queryBuilder = queryBuilder.eq("is_spoof", true);
    dataLabel = "Proxy Attempts";
    filterApplied = true;
  }

  // B. Check for INTRUDERS
  if (
    userQuestion.toLowerCase().includes("intruder") ||
    userQuestion.toLowerCase().includes("intruders")
  ) {
    // Intruder logs are logged as is_spoof=FALSE but roll_no=NULL
    queryBuilder = queryBuilder.is("roll_no", null);
    dataLabel = "Intruder Sightings";
    filterApplied = true;
  }

  // C. Check for Specific Student (e.g., "Ashraf")
  if (
    userQuestion.toLowerCase().includes("ashraf") ||
    userQuestion.toLowerCase().includes("mahaboob")
  ) {
    // NOTE: This is a simple heuristic. A better system would search the students table first.
    const namePart = userQuestion.toLowerCase().includes("ashraf")
      ? "ashraf"
      : "mahaboob";

    // Fetch the target student's full data (assuming roll_no lookup exists in students table)
    // For simplicity, we will query logs where the student eventually logged in.
    const { data: studentData } = await supabaseClient
      .from("students")
      .select("roll_no")
      .ilike("full_name", `%${namePart}%`);

    if (studentData && studentData.length > 0) {
      const roll_nos = studentData.map((s) => s.roll_no);
      queryBuilder = queryBuilder.in("roll_no", roll_nos);
      dataLabel = `${studentData[0].full_name}'s Activity`;
      filterApplied = true;
    } else if (!filterApplied) {
      return `AI Assistant: I cannot find activity logs for '${namePart}'. Please check the name or Roll Number.`;
    }
  }

  // D. Time Constraint: Default to today if 'today' is mentioned or no other filters are present
  if (!filterApplied || userQuestion.toLowerCase().includes("today")) {
    queryBuilder = queryBuilder.gte("timestamp", today); // gte = greater than or equal to
    dataLabel += " (Today)";
  }

  // Final Data Retrieval
  const { data: rawData, error: dbError } = await queryBuilder.order(
    "timestamp",
    { ascending: false }
  );

  if (dbError) {
    console.error("Supabase Query Error:", dbError);
    return "AI Assistant: A database error occurred during data retrieval.";
  }

  if (!rawData || rawData.length === 0) {
    return `AI Assistant: No ${dataLabel} found in the logs matching your criteria.`;
  }

  // --- STEP 2: AUGMENTATION & STEP 3: GENERATION ---

  const userQueryWithContext = `User Question: "${userQuestion}". Data Retrieved (${
    rawData.length
  } records for ${dataLabel}): ${JSON.stringify(rawData)}`;

  console.log(`[RAG] Sending query to Gemini for analysis on ${dataLabel}...`);

  // Call the core API function
  const aiResponse = await generateContent(
    ANALYST_PROMPT,
    userQueryWithContext
  );

  return `AI Assistant: ${aiResponse}`;
}
