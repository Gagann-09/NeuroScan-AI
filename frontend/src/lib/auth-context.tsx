"use client";
import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { onAuthStateChanged, User, signOut } from "firebase/auth";
import { auth } from "./firebase";

interface AuthContextType {
    user: User | null;
    loading: boolean;
    getIdToken: () => Promise<string | null>;
    logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
    user: null,
    loading: true,
    getIdToken: async () => null,
    logout: async () => {},
});

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);

    const getIdToken = useCallback(async () => {
        if (!user) return null;
        try {
            return await user.getIdToken(true); // force refresh
        } catch (err) {
            console.error("Failed to get ID token:", err);
            return null;
        }
    }, [user]);

    const logout = useCallback(async () => {
        try {
            await signOut(auth);
        } catch (err) {
            console.error("Sign out error:", err);
        }
    }, []);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
            setLoading(false);
        });
        return () => unsubscribe();
    }, []);

    return (
        <AuthContext.Provider value={{ user, loading, getIdToken, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);