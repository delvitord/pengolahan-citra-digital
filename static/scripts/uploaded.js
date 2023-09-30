document.addEventListener("DOMContentLoaded", function () {
    const rgbTableBody = document.querySelector("#rgb-table tbody");
    const showRgbTableButton = document.getElementById("showRgbTableButton");
    const rgbTable = document.getElementById("rgb-table");

    var image = document.getElementById('image-result');

    // Menggunakan fungsi naturalWidth dan naturalHeight untuk mendapatkan dimensi gambar asli
    var width = image.naturalWidth;
    var height = image.naturalHeight;
  
    // Menampilkan dimensi di dalam elemen p dengan id "image-dimensions"
    var dimensionsElement = document.getElementById('image-dimensions');
    dimensionsElement.textContent = 'Lebar: ' + width + 'px, Tinggi: ' + height + 'px';

    showRgbTableButton.addEventListener("click", function () {
        // Toggle the visibility of the RGB table
        rgbTable.classList.toggle("d-none");

        if (!rgbTable.classList.contains("d-none")) {
            // Get RGB values from the image when the table is visible
            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            context.drawImage(image, 0, 0, image.width, image.height);
            const imageData = context.getImageData(0, 0, image.width, image.height).data;

            // Check if the image is grayscale
            let isGrayscale = true;
            for (let i = 0; i < imageData.length; i += 4) {
                const r = imageData[i];
                const g = imageData[i + 1];
                const b = imageData[i + 2];
                if (r !== g || g !== b) {

                    isGrayscale = false;
                    break;
                }
            }

            // Update the RGB table with values or display a message for grayscale images
            if (isGrayscale) {
                rgbTableBody.innerHTML = "";
                rgbTableBody.innerHTML = `<tr><td colspan="3">This is a grayscale image.</td></tr>`;
            } else {
                rgbTableBody.innerHTML = "";
                for (let i = 0; i < imageData.length; i += 4) {
                    const r = imageData[i];
                    const g = imageData[i + 1];
                    const b = imageData[i + 2];

                    const row = document.createElement("tr");
                    row.innerHTML = `<td>${r}</td><td>${g}</td><td>${b}</td>`;
                    rgbTableBody.appendChild(row);
                }
            }
        }
    });
});

    let rangeMin = 0;
    const range = document.querySelector(".range-selected");
    const rangeInput = document.querySelectorAll(".range-input input");
    const rangethres = document.querySelectorAll(".range-thres input");

    rangeInput.forEach((input) => {
        input.addEventListener("input", (e) => {
          let minRange = parseInt(rangeInput[0].value);
          let maxRange = parseInt(rangeInput[1].value);
          if (maxRange - minRange < rangeMin) {     
            if (e.target.className === "min") {
              rangeInput[0].value = maxRange - rangeMin;        
            } else {
              rangeInput[1].value = minRange + rangeMin;        
            }
          } else {
            rangethres[0].value = minRange;
            rangethres[1].value = maxRange;
            range.style.left = (minRange / rangeInput[0].max) * 100 + "%";
            range.style.right = 100 - (maxRange / rangeInput[1].max) * 100 + "%";
          }
        });
      });

      rangethres.forEach((input) => {
        input.addEventListener("input", (e) => {
          let minthres = rangethres[0].value;
          let maxthres = rangethres[1].value;
          if (maxthres - minthres >= rangeMin && maxthres <= rangeInput[1].max) {
            if (e.target.className === "min") {
              rangeInput[0].value = minthres;
              range.style.left = (minthres / rangeInput[0].max) * 100 + "%";
            } else {
              rangeInput[1].value = maxthres;
              range.style.right = 100 - (maxthres / rangeInput[1].max) * 100 + "%";
            }
          }
        });
      });

      // Seleksi elemen input tipe "number"
const lowerThresInput = document.getElementById("lower_thres");
const upperThresInput = document.getElementById("upper_thres");

// Tambahkan event listener ke input tipe "number" untuk Batas Awal
lowerThresInput.addEventListener("input", () => {
    // Perbarui nilai slider sesuai dengan nilai input
    rangeInput[0].value = lowerThresInput.value;
    // Perbarui tampilan slider
    range.style.left = (lowerThresInput.value / rangeInput[0].max) * 100 + "%";
});

// Tambahkan event listener ke input tipe "number" untuk Batas Akhir
upperThresInput.addEventListener("input", () => {
    // Perbarui nilai slider sesuai dengan nilai input
    rangeInput[1].value = upperThresInput.value;
    // Perbarui tampilan slider
    range.style.right = 100 - (upperThresInput.value / rangeInput[1].max) * 100 + "%";
});


const imageList = document.getElementById("image-list");
const imgNormal = "img_normal.jpg"; // Gambar normal awal

// Fungsi untuk menambahkan gambar ke daftar riwayat
function addToImageHistory(imageName) {
    const listItem = document.createElement("li");
    listItem.textContent = imageName;
    imageList.appendChild(listItem);
}

// Menerima informasi perubahan gambar dari Server-Sent Events
const eventSource = new EventSource("/image_changes");

eventSource.addEventListener("message", function (event) {
    const imageName = event.data;
    addToImageHistory(imageName);
});

// Dapatkan elemen gambar riwayat
const historyImages = document.querySelectorAll('.history-image');
// Dapatkan elemen gambar yang akan diperbarui
const afterImage = document.getElementById('image-result');

// Tambahkan event listener untuk setiap gambar riwayat
historyImages.forEach(image => {
    image.addEventListener('click', () => {
        // Dapatkan nomor gambar dari atribut data-img-num
        const imgNum = image.getAttribute('data-img-num');
        // Perbarui gambar yang ditampilkan di "After" sesuai dengan yang diklik
        afterImage.src = `/static/img/img${imgNum}.jpg`;
    });
});

// Dapatkan elemen form history
const restoreForm = document.getElementById('restore-form');
// Tambahkan event listener untuk setiap gambar riwayat
historyImages.forEach(image => {
    image.addEventListener('click', () => {
        // Dapatkan nomor gambar dari atribut data-img-num
        const imgNum = image.getAttribute('data-img-num');
        // Perbarui nilai atribut value pada tombol "Restore" di dalam form history
        // Perbarui teks pada tombol "Restore" di dalam form history
        document.getElementById('image-counter').textContent = imgNum;
        restoreForm.elements.restore_img.value = imgNum;
        restoreForm.action = `/restore_history/${imgNum}`;
    });
});
