import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDfVTinwQuShIGsZVGZKww-zyAS0BmxKHs",
  authDomain: "neuroscan-medical-vault.firebaseapp.com",
  projectId: "neuroscan-medical-vault",
  storageBucket: "neuroscan-medical-vault.firebasestorage.app",
  messagingSenderId: "459273125762",
  appId: "1:459273125762:web:b9b23c8617f72051b2599b"
};

// Initialize the Bouncer
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

