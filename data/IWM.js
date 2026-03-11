let csvContent = "";

// find the historical data table
const table = document.querySelector("table");

// get headers
const headers = [...table.querySelectorAll("thead th")]
  .map(th => th.textContent.trim().replace(/,/g, ""));

csvContent += headers.join(",") + "\n";

// get rows
table.querySelectorAll("tbody tr").forEach(row => {
  const cols = [...row.querySelectorAll("td")]
    .map(td => td.textContent.trim().replace(/,/g, ""));

  if (cols.length === headers.length) {
    csvContent += cols.join(",") + "\n";
  }
});

// create and download CSV
const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
const url = URL.createObjectURL(blob);

const link = document.createElement("a");
link.href = url;
link.download = "IWM_yahoo_historical_data.csv";

document.body.appendChild(link);
link.click();
document.body.removeChild(link);