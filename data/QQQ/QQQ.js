let csvContent = "date,close\n";

document.querySelector("table tbody")
  .querySelectorAll("tr")
  .forEach((row) => {

    if (row.cells.length < 6) return; // skip dividend rows

    const date = row.cells[0].textContent.trim();
    const close = row.cells[4].textContent.trim();

    const price = parseFloat(close.replace(',', ''));

    if (!isNaN(price)) {
      csvContent += `${date},${price}\n`;
    }

});

// create file and download
const blob = new Blob([csvContent], { type: "text/csv" });
const url = URL.createObjectURL(blob);

const link = document.createElement("a");
link.href = url;
link.download = "QQQ_prices.csv";
document.body.appendChild(link);
link.click();

document.body.removeChild(link);
URL.revokeObjectURL(url);