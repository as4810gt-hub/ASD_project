(() => {
    const toast = document.querySelector("#app-toast");

    window.showToast = (message, tone = "default") => {
        if (!toast) return;
        toast.textContent = message;
        toast.dataset.tone = tone;
        toast.classList.add("is-visible");
        window.clearTimeout(window.__toastTimer);
        window.__toastTimer = window.setTimeout(() => {
            toast.classList.remove("is-visible");
        }, 2800);
    };
})();

