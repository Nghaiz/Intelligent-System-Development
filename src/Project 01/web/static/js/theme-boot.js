// Đặt nền trước khi vẽ trang đầu để không chớp nền sai.
try {
  const saved = localStorage.getItem("dnx-theme");
  const light = window.matchMedia("(prefers-color-scheme: light)").matches;
  document.documentElement.dataset.theme = saved || (light ? "light" : "dark");
} catch { /* localStorage bị chặn: giữ nền tối mặc định */ }
