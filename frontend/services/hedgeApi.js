const API_BASE_URL = "http://localhost:8000";

export async function getHedgeRecommendation(payload) {
  const response = await fetch(`${API_BASE_URL}/api/hedge/recommend`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction request failed");
  }

  return data;
}
