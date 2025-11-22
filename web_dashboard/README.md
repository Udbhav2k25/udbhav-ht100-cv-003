# Web Dashboard (Single-file React App)

This folder contains a single-file React application (`src/App.jsx`) that connects directly to your Supabase database and displays a dark-themed, real-time attendance & security dashboard.

Quick setup (PowerShell)

```powershell
cd C:\Users\786\Desktop\hack\web_dashboard
npm install
npm run dev
```

Environment variables

- Create a `.env` file in `web_dashboard` with the following values (Vite will expose `VITE_` vars to the client):

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_KEY=your-anon-or-service-key
```

Notes

- The app expects two tables in your Supabase schema:
  - `students(roll_no, full_name)`
  - `attendance_logs(roll_no, event_type, timestamp, is_spoof, evidence_url)`
- The dashboard subscribes to INSERT events on `attendance_logs` and updates live.
- For quick MVP styling Tailwind is loaded via CDN in `index.html`. For production switch to a proper Tailwind build process.
- Using an anon key in the browser exposes it publicly; use RLS policies on Supabase to restrict access or proxy requests through a backend for stricter security.

# React Dashboard.
