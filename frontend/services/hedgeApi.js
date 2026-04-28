const API_BASE_URL = "http://localhost:8001";

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

export async function getQuote(ticker) {
  const response = await fetch(
    `${API_BASE_URL}/api/quote?ticker=${encodeURIComponent(ticker)}`,
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Quote request failed");
  }
  return data;
}

export async function getHistory(ticker, window = 15) {
  const response = await fetch(
    `${API_BASE_URL}/api/history?ticker=${encodeURIComponent(ticker)}&window=${window}`,
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "History request failed");
  }
  return data;
}

export async function placePaperOrder(payload) {
  const response = await fetch(`${API_BASE_URL}/api/orders/paper`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Paper order failed");
  }
  return data;
}

export async function getPaperOrders(limit = 20) {
  const response = await fetch(
    `${API_BASE_URL}/api/orders/paper?limit=${encodeURIComponent(limit)}`,
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not load paper orders");
  }
  return data;
}
