const uploadBox = document.querySelector('.upload-box');
const fileInput = document.getElementById('csvFileInput');
const csvTextOutput = document.getElementById('csv-output-text');
const csvTableOutput = document.getElementById('csv-output-table');

const btnText = document.getElementById('btn-text');
const btnTable = document.getElementById('btn-table');
const analyzeBtn = document.getElementById('analyzeBtn');       // ✅ Run Analysis 버튼
const uploadForm = document.getElementById('uploadForm');       // ✅ 숨겨진 form

// 파일 업로드 박스 클릭 → 파일 선택창 열기
uploadBox.addEventListener('click', () => {
  fileInput.click();
});

// 파일 선택 완료 시 미리보기 수행
fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;

    // 표 형태 먼저 생성
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');

    let tableHtml = '<table border="1" cellpadding="6" style="border-collapse:collapse;">';
    tableHtml += '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>';
    tableHtml += '<tbody>';
    for (let i = 1; i < lines.length; i++) {
      const cells = lines[i].split(',');
      tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
    }
    tableHtml += '</tbody></table>';

    // 텍스트/표 미리보기 표시
    csvTextOutput.textContent = text;
    csvTableOutput.innerHTML = tableHtml;

    csvTextOutput.style.display = 'none';
    csvTableOutput.style.display = 'block';
  };

  reader.readAsText(file);
});

// Run Analysis 버튼 누르면 서버에 form 제출
analyzeBtn.addEventListener('click', () => {
  if (!fileInput.files[0]) {
    alert("⚠️ CSV 파일을 먼저 선택하세요.");
    return;
  }

  uploadForm.submit();  // ✅ form POST 제출
});

// 텍스트 보기 버튼
btnText.addEventListener('click', () => {
  csvTextOutput.style.display = 'block';
  csvTableOutput.style.display = 'none';
});

// 표 보기 버튼
btnTable.addEventListener('click', () => {
  csvTextOutput.style.display = 'none';
  csvTableOutput.style.display = 'block';
});

// 페이지 진입 시 서버에서 온 테이블이 있으면 자동 표시
window.addEventListener('DOMContentLoaded', () => {
  const preText = document.getElementById('csv-output-text').textContent;
  if (preText.trim() !== "") {
    const lines = preText.trim().split('\n');
    const headers = lines[0].split(',');

    let tableHtml = '<table class="csv-table">';

    tableHtml += '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>';
    tableHtml += '<tbody>';
    for (let i = 1; i < lines.length; i++) {
      const cells = lines[i].split(',');
      tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
    }
    tableHtml += '</tbody></table>';

    // 텍스트/표 미리보기 표시
    document.getElementById('csv-output-table').innerHTML = tableHtml;
    document.getElementById('csv-output-text').style.display = 'none';
    document.getElementById('csv-output-table').style.display = 'block';
  }
});
