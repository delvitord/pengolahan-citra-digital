document.getElementById("uploadForm").addEventListener("submit", function(event) {
    var fileInput = document.getElementById("inputGroupFile04");
    if (fileInput.files.length === 0) {
        alert("Pilih gambar terlebih dahulu.");
        event.preventDefault(); // Mencegah pengiriman formulir
    }
});


var dropArea = document.getElementById("dropArea");
var dropText = document.getElementById("dropText");
var uploadStatus = document.getElementById("uploadStatus");

// Mencegah tindakan default saat file dijatuhkan
dropArea.addEventListener("dragenter", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.add("highlight");
    dropText.innerText = "Lepaskan saja!";
}, false);

dropArea.addEventListener("dragleave", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.remove("highlight");
    dropText.innerText = "Lempar sini citranya atau klik untuk memilih";
}, false);

dropArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    e.stopPropagation();
}, false);

dropArea.addEventListener("drop", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.remove("highlight");
    dropText.innerText = "Lempar sini citranya atau klik untuk memilih";

    var fileInput = document.getElementById("inputGroupFile04");
    var files = e.dataTransfer.files;

    if (files.length > 0) {
        fileInput.files = files;
        updateUploadStatus(files[0].name);
    }
}, false);

// Menangani klik pada area drag and drop
dropArea.addEventListener("click", function () {
    var fileInput = document.getElementById("inputGroupFile04");
    fileInput.click();
});

// Menangani event saat file dipilih
var fileInput = document.getElementById("inputGroupFile04");
fileInput.addEventListener("change", function () {
    var fileName = fileInput.files[0] ? fileInput.files[0].name : null;
    updateUploadStatus(fileName);
});

function updateUploadStatus(fileName) {
    if (fileName) {
        uploadStatus.innerText = `${fileName} berhasil diunggah.`;
    } else {
        uploadStatus.innerText = "Anda belum memasukkan citra.";
    }
}

// Mengecek status pada awal load
updateUploadStatus(null); // Menampilkan pesan "Anda belum memasukkan citra."
