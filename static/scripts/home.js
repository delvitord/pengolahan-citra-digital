document.getElementById("uploadForm").addEventListener("submit", function(event) {
    var fileInput = document.getElementById("inputGroupFile04");
    if (fileInput.files.length === 0) {
        alert("Pilih gambar terlebih dahulu.");
        event.preventDefault(); // Mencegah pengiriman formulir
    }
});

