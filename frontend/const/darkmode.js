const themeSwitch = document.getElementById('theme-switch');
const body = document.body;

// Check if the user has a preferred theme stored
if (localStorage.getItem('theme') === 'dark') {
  body.classList.add('dark-mode');
  themeSwitch.checked = true;
  document.getElementById('theme-label').textContent = 'Light Mode';
} else {
  document.getElementById('theme-label').textContent = 'Dark Mode';
}

// Toggle theme on change
themeSwitch.addEventListener('change', () => {
  if (themeSwitch.checked) {
    body.classList.add('dark-mode');
    localStorage.setItem('theme', 'dark');
    document.getElementById('theme-label').textContent = 'Light Mode';
  } else {
    body.classList.remove('dark-mode');
    localStorage.setItem('theme', 'light');
    document.getElementById('theme-label').textContent = 'Dark Mode';
  }
});
