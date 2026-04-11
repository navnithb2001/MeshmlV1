# MeshML Dashboard Firebase Deployment Guide

This document details the one-time setup and ongoing deployment workflow to host the React/Vite dashboard statically on Firebase Hosting.

## One-Time Initialization

Before using the automated deployment script for the first time, you must initialize the Firebase project within the `dashboard` directory.

### 1. Install Firebase CLI
Install the tools globally via npm:
```bash
npm install -g firebase-tools
```

### 2. Login
Authenticate the CLI with your Google account:
```bash
firebase login
```

### 3. Initialize Hosting
Navigate into the dashboard directory and initialize the project:
```bash
cd dashboard
firebase init hosting
```

**When prompted by the Firebase CLI, provide the following specific answers:**
* **What do you want to use as your public directory?** `dist` *(This is where Vite outputs the production build)*
* **Configure as a single-page app (rewrite all urls to /index.html)?** `Yes` *(Required for React Router to function properly)*
* **Set up automatic builds and deploys with GitHub?** `No` *(Optional - answer based on your preference)*

---

## Automated Deployment

Once initialized, you can use the bespoke bash script to build and sync all changes continuously.

From the root of the MeshML repository:

1. Make the script executable:
```bash
chmod +x scripts/deploy-dashboard.sh
```

2. Run the deployment:
```bash
./scripts/deploy-dashboard.sh
```

### What the script does:
1. Navigates into the `dashboard/` directory.
2. Ensures all dependencies are synced with `npm install`.
3. Compiles a highly-optimized production bundle into the `dist/` folder via `npm run build`.
4. Pushes the static assets securely to Google's global CDN edge network using `firebase deploy --only hosting`.
