// Contact Popup
const contactBtn = document.getElementById('contact-btn');
const contactPopup = document.getElementById('contact-popup');
const closePopup = document.getElementById('close-popup');

if (contactBtn && contactPopup && closePopup) {
    contactBtn.addEventListener('click', () => {
        contactPopup.style.display = 'flex';
    });

    closePopup.addEventListener('click', () => {
        contactPopup.style.display = 'none';
    });
}

// Slideshow
const slides = document.querySelectorAll('.slides img');
let currentSlide = 0;

function showSlide(index) {
    slides.forEach((slide, i) => {
        slide.classList.toggle('active', i === index);
    });
}

function nextSlide() {
    currentSlide = (currentSlide + 1) % slides.length;
    showSlide(currentSlide);
}

if (slides.length > 0) {
    showSlide(currentSlide);
    setInterval(nextSlide, 3000);
}

// Firebase Functionality
document.addEventListener('DOMContentLoaded', () => {
    if (typeof firebase === 'undefined') {
        console.error('Firebase SDK not loaded');
        return;
    }
    console.log('Firebase SDK detected');

    const auth = firebase.auth();
    const db = firebase.firestore();

    // Password Validation
    function validatePassword(password) {
        const numberCount = (password.match(/\d/g) || []).length;
        const hasUpperCase = /[A-Z]/.test(password);
        const hasLowerCase = /[a-z]/.test(password);
        const hasSpecialChar = /[@\-_]/.test(password);
        return numberCount >= 5 && hasUpperCase && hasLowerCase && hasSpecialChar;
    }

    // Toggle Password Visibility
    function togglePassword(inputId, toggleId) {
        const input = document.getElementById(inputId);
        const toggle = document.getElementById(toggleId);
        toggle.addEventListener('click', () => {
            const type = input.type === 'password' ? 'text' : 'password';
            input.type = type;
            toggle.textContent = type === 'password' ? '👁️' : '👁️‍🗨️';
        });
    }

    togglePassword('login-password', 'toggle-login-password');
    togglePassword('password', 'toggle-create-password');

    // Password Strength Bar
    const passwordInput = document.getElementById('password');
    const strengthBar = document.getElementById('password-strength');
    if (passwordInput && strengthBar) {
        passwordInput.addEventListener('input', () => {
            const password = passwordInput.value;
            let strength = 0;
            if (password.length > 0) strength += 20;
            if ((password.match(/\d/g) || []).length >= 5) strength += 20;
            if (/[A-Z]/.test(password)) strength += 20;
            if (/[a-z]/.test(password)) strength += 20;
            if (/[@\-_]/.test(password)) strength += 20;

            strengthBar.style.width = `${strength}%`;
            if (strength <= 40) {
                strengthBar.style.background = '#ff3333';
            } else if (strength <= 80) {
                strengthBar.style.background = '#f88f09';
            } else {
                strengthBar.style.background = '#216e89';
            }
        });
    }

    // Form Switching
    const loginSection = document.getElementById('login-section');
    const createSection = document.getElementById('create-account-section');
    const createLink = document.getElementById('create-account-link');
    const backToLoginLink = document.getElementById('back-to-login-link');

    createLink.addEventListener('click', (e) => {
        e.preventDefault();
        loginSection.classList.add('hidden');
        createSection.classList.remove('hidden');
    });

    backToLoginLink.addEventListener('click', (e) => {
        e.preventDefault();
        createSection.classList.add('hidden');
        loginSection.classList.remove('hidden');
    });

    // Create Account
    const createForm = document.getElementById('create-form');
    if (createForm) {
        console.log('Create form element found');
        createForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('Create form submitted');

            const name = document.getElementById('name').value.trim();
            const surname = document.getElementById('surname').value.trim();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            const error = document.getElementById('create-error');

            error.textContent = '';

            if (!name || !surname || !email) {
                error.textContent = 'All fields are required.';
                console.log('Validation failed: Missing fields');
                return;
            }
            if (!validatePassword(password)) {
                error.textContent = 'Password must have at least 5 numbers, uppercase, lowercase, and special characters (@-_).';
                console.log('Validation failed: Weak password');
                return;
            }
            if (password !== confirmPassword) {
                error.textContent = 'Passwords do not match.';
                console.log('Validation failed: Password mismatch');
                return;
            }

            try {
                console.log('Attempting Firebase user creation');
                const userCredential = await auth.createUserWithEmailAndPassword(email, password);
                const user = userCredential.user;
                console.log('User created:', user.uid);

                await user.sendEmailVerification();
                console.log('Verification email sent to:', email);

                document.getElementById('create-form').classList.add('hidden');
                document.getElementById('verify-section').classList.remove('hidden');
                backToLoginLink.classList.add('hidden');

                const checkVerification = setInterval(async () => {
                    await user.reload();
                    if (user.emailVerified) {
                        console.log('Email verified for:', email);
                        clearInterval(checkVerification);

                        await db.collection('users').doc(user.uid).set({
                            name: name,
                            surname: surname,
                            email: email,
                            createdAt: firebase.firestore.FieldValue.serverTimestamp()
                        }, { merge: true });
                        console.log('User details stored in Firestore:', user.uid);

                        const idToken = await user.getIdToken();
                        const response = await fetch('/login', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ idToken })
                        });
                        const result = await response.json();
                        if (result.success) {
                            window.location.href = result.redirect;
                        }
                    }
                }, 2000);

                auth.onAuthStateChanged((currentUser) => {
                    if (!currentUser || (currentUser && currentUser.emailVerified)) {
                        clearInterval(checkVerification);
                    }
                });
            } catch (err) {
                console.error('Create account error:', err.code, err.message);
                error.textContent = err.message || 'An error occurred. Please try again.';
                document.getElementById('create-form').classList.remove('hidden');
                document.getElementById('verify-section').classList.add('hidden');
            }
        });
    }

    // Login
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {  // Added async here
            e.preventDefault();
            console.log('Login form submitted');
            const email = document.getElementById('login-email').value.trim();
            const password = document.getElementById('login-password').value;
            const error = document.getElementById('login-error');

            error.textContent = '';

            try {
                const userCredential = await auth.signInWithEmailAndPassword(email, password);
                const user = userCredential.user;
                if (!user.emailVerified) {
                    error.textContent = 'Please verify your email before logging in.';
                    await auth.signOut();
                    return;
                }
                console.log('User logged in:', user.uid);
                const idToken = await user.getIdToken();
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ idToken })
                });
                const result = await response.json();
                if (result.success) {
                    window.location.href = result.redirect;
                } else {
                    error.textContent = result.message;
                }
            } catch (err) {
                console.error('Login error:', err.code, err.message);
                error.textContent = 'Invalid email or password.';
            }
        });
    }

    // Forgot Password
    const forgotPasswordLink = document.getElementById('forgot-password-link');
    const sendResetCode = document.getElementById('send-reset-code');
    if (forgotPasswordLink && sendResetCode) {
        forgotPasswordLink.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('forgot-password').classList.remove('hidden');
        });

        sendResetCode.addEventListener('click', async () => {
            const email = document.getElementById('forgot-email').value.trim();
            const error = document.getElementById('forgot-error');

            error.textContent = '';

            try {
                console.log('Sending password reset email to:', email);
                await auth.sendPasswordResetEmail(email);
                error.textContent = 'A password reset link has been sent to your email.';
                error.style.color = '#216e89';
            } catch (err) {
                console.error('Reset password error:', err.code, err.message);
                error.textContent = 'Email not found or an error occurred.';
            }
        });
    }
});