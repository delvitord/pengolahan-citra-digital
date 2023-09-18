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
var imagePreview = document.getElementById("imagePreview");
var dragging = false;

// Mencegah tindakan default saat file dijatuhkan
dropArea.addEventListener("dragenter", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.add("highlight");
    dropText.innerText = "Lepaskan saja!";
    dragging = true;
}, false);

dropArea.addEventListener("dragleave", function (e) {
    e.preventDefault();
    e.stopPropagation();
    if (!dragging) {
        dropArea.classList.remove("highlight");
        dropText.innerText = "Lempar sini citranya atau klik untuk memilih";
    }
    dragging = false;
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
    dragging = false;

    var fileInput = document.getElementById("inputGroupFile04");
    var files = e.dataTransfer.files;

    if (files.length > 0) {
        fileInput.files = files;
        updateUploadStatus(files[0].name);
        displayImagePreview(files[0]);
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
    if (fileInput.files[0]) {
        displayImagePreview(fileInput.files[0]);
    } else {
        imagePreview.style.display = "none";
    }
});

function updateUploadStatus(fileName) {
    if (fileName) {
        uploadStatus.innerText = `${fileName} berhasil diunggah.`;
    } else {
        uploadStatus.innerText = "Anda belum memasukkan citra.";
    }
}

function displayImagePreview(file) {
    var reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
    };
    reader.readAsDataURL(file);
}

// Mengecek status pada awal load
updateUploadStatus(null); // Menampilkan pesan "Anda belum memasukkan citra."