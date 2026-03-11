const tables = document.querySelectorAll("table");

if (!tables.length) {
  console.error("No tables found on this page.");
} else {

  tables.forEach((table, index) => {

    let csv = "";
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
    link.setAttribute("download", `DIA_options_table_${index+1}.csv`);

    document.body.appendChild(link);
    link.click();

    document.body.removeChild(link);
    URL.revokeObjectURL(url);

  });

  console.log("DIA options CSV files downloaded.");
}