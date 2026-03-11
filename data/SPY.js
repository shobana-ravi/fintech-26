let csv = "";

// find the main Yahoo Finance data table
const table = document.querySelector("table[data-test='option-chain-table'], table");

if (!table) {
  console.error("No table found on this page.");
} else {

  const rows = table.querySelectorAll("tr");

  // headers
  const headers = [...rows[0].querySelectorAll("th")]
    .map(h => h.innerText.trim());

  csv += headers.join(",") + "\n";

  // rows
  rows.forEach((row, i) => {
    if (i === 0) return;

    const cols = [...row.querySelectorAll("td")]
      .map(td => td.innerText.trim().replace(/,/g,""));

    if (cols.length) csv += cols.join(",") + "\n";
  });

  // create CSV file
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });

  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);

  link.setAttribute("href", url);
  link.setAttribute("download", "SPY_data.csv");

  document.body.appendChild(link); // required for some browsers
  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  console.log("SPY_data.csv downloaded.");
}