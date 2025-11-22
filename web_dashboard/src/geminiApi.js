// web_dashboard/src/geminiApi.js

// 1. Configuration: Load Key from React's environment variables
// Ensure this variable is set in web_dashboard/.env.local (VITE_GEMINI_API_KEY="")
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "";
const API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/";
const MODEL = "gemini-2.5-flash-preview-09-2025";

// --- Core API Function ---

/**
 * Calls the Gemini API with a system prompt and a content prompt.
 * Implements exponential backoff for reliability.
 * * @param {string} systemInstruction - The persona/role for the model.
 * @param {string} userQuery - The specific question and context (data) to analyze.
 * @param {number} maxRetries - Max number of retries for backoff.
 * @returns {Promise<string>} The generated text response from Gemini.
 */
export async function generateContent(
  systemInstruction,
  userQuery,
  maxRetries = 3
) {
  if (!API_KEY) {
    console.error("Gemini API Key is missing. Please set VITE_GEMINI_API_KEY.");
    return "Error: AI Assistant is offline (Missing API Key).";
  }

  const apiUrl = `${API_URL_BASE}${MODEL}:generateContent?key=${API_KEY}`;
  const payload = {
    contents: [{ parts: [{ text: userQuery }] }],
    // Setting a clear System Instruction for the AI Analyst persona
    systemInstruction: {
      parts: [{ text: systemInstruction }],
    },
  };

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.status === 429) {
        // Too Many Requests (Rate Limit) -> Wait and retry
        const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
        console.warn(
          `Attempt ${attempt + 1}: Rate limit hit. Retrying in ${Math.round(
            delay / 1000
          )}s...`
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(
          `API Error ${response.status}: ${JSON.stringify(errorBody)}`
        );
      }

      const result = await response.json();
      const generatedText = result.candidates?.[0]?.content?.parts?.[0]?.text;

      if (generatedText) {
        return generatedText;
      } else {
        throw new Error("Received empty response from the API.");
      }
    } catch (error) {
      console.error(`Gemini API Failed on attempt ${attempt + 1}:`, error);
      if (attempt === maxRetries - 1) {
        return "Error: AI Assistant failed to process the request after multiple retries.";
      }
      // Wait before retrying non-rate-limit errors
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
  return "Error: AI Assistant failed to process the request.";
}

// Example usage to be used in the main React component:
/*
async function runAnalysis(rawData) {
    const system = "You are a professional security analyst. Your response must be concise and based only on the provided JSON data.";
    const query = `Analyze the following attendance logs and report the total number of security threats. Logs: ${JSON.stringify(rawData)}`;
    const result = await generateContent(system, query);
    console.log(result);
}
*/
