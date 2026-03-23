const fs = require("fs");

// Node 18+ has built-in fetch
// If older version, install: npm install node-fetch

const tickers = ["aapl.us", "msft.us", "tsla.us"];

async function downloadData() {
  for (const ticker of tickers) {
    const url = `https://stooq.com/q/d/l/?s=${ticker}&i=d`;

    try {
      const response = await fetch(url);
      const data = await response.text();

      // Save directly as CSV
      const fileName = `data/${ticker}.csv`;

      fs.writeFileSync(fileName, data);

      console.log(`${ticker} saved to ${fileName}`);
    } catch (err) {
      console.error(`Error with ${ticker}:`, err);
    }
  }
}

downloadData();