# Guide: Automated Daily Yelp Labeling with 5 Groq Keys (GitHub Actions)

This guide walks you through setting up the automated daily labeling pipeline using GitHub Actions (100% free, runs in the cloud with no local PC required).

---

## 1. Why GitHub's 6-Hour Limit Won't Be an Issue

You mentioned experiencing an action stopping after 6 hours in the past. Here is why this pipeline will **never** hit that limit:

- **GitHub's 6-Hour Rule**: GitHub kills any single workflow job that runs continuously for 360 minutes (6 hours). This usually happens when an action is stuck in an infinite loop or a long-running compile.
- **How Our Pipeline Works**:
  - Each Groq key on the free tier has a daily limit (e.g. 1,000 requests/day or 100k-500k tokens/day).
  - Groq operates fast: at 15–20 requests per minute, a key exhausts its daily quota in **~15 to 25 minutes**.
  - Across all 5 keys, the daily batch finishes in **~1 to 1.5 hours total**.
  - As soon as the 5th key hits its daily limit, the script logs:  
    `"All 5 Groq keys exhausted for today. Checkpoint saved. Exiting."`  
    and immediately **exits with code 0**.
  - GitHub Actions then commits the newly labeled data back to your repo and **ends the job**.
  - The job only runs for ~1 to 1.5 hours per day (well within GitHub's 6-hour limit).
  - The next day at `00:05 UTC`, GitHub's cron triggers a **brand new fresh job**.
  - As an extra precaution, we also configured `timeout-minutes: 240` (4 hours) and a built-in Python watchdog timer to guarantee the job always exits cleanly.

---

## 2. Step 1: Initialize Git and Push to GitHub

Since your project folder is not yet a Git repository, initialize it and push to GitHub:

```bash
# 1. Initialize git
git init -b main

# 2. Add files (the .gitignore we created automatically excludes >100MB raw files and .env)
git add .

# 3. Create your initial commit
git commit -m "feat: setup ABSA research pipeline and daily Groq labeling action"

# 4. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git

# 5. Push to GitHub
git push -u origin main
```

> [!NOTE]
> The `.gitignore` file we added prevents the 5.3 GB raw Yelp review file from being uploaded, keeping your repository lightweight and fast. The sampled reviews file (`data/raw/yelp_restaurants_5000.jsonl` or `yelp_restaurants_sampled.jsonl`) will be uploaded.

---

## 3. Step 2: Add Your 5 Groq API Keys to GitHub Secrets

1. Go to your repository on GitHub.
2. Click **Settings** (tab at the top right) → **Secrets and variables** (in the left sidebar) → **Actions**.
3. Click the green **New repository secret** button.
4. You can add them in either of these two ways:

### Option A (Single Secret - Recommended):
- **Name**: `GROQ_API_KEYS`
- **Secret**: Paste all 5 keys separated by commas:  
  `gsk_key1...,gsk_key2...,gsk_key3...,gsk_key4...,gsk_key5...`

### Option B (Individual Secrets):
Add 5 separate secrets:
- `GROQ_API_KEY_1`: `gsk_...`
- `GROQ_API_KEY_2`: `gsk_...`
- `GROQ_API_KEY_3`: `gsk_...`
- `GROQ_API_KEY_4`: `gsk_...`
- `GROQ_API_KEY_5`: `gsk_...`

*(The code automatically checks and supports both options!)*

---

## 4. Step 3: Enable Workflow Read & Write Permissions

In order for GitHub Actions to commit the newly labeled reviews back into your repository:

1. In your GitHub repo, go to **Settings** → **Actions** → **General**.
2. Scroll down to **Workflow permissions**.
3. Select **Read and write permissions**.
4. Check the box **Allow GitHub Actions to create and approve pull requests** (optional, but good to have).
5. Click **Save**.

---

## 5. Step 4: Testing & Monitoring

### Trigger Manually Anytime:
You don't need to wait for midnight UTC to test it!
1. Go to the **Actions** tab in your GitHub repository.
2. Click on **Daily Yelp Review Labeler (Groq 5-Key)** in the left sidebar.
3. Click the **Run workflow** dropdown on the right and click **Run workflow**.
4. Watch the live console output as it rotates through your keys and commits the labeled results.

### Automatic Schedule:
- The workflow will automatically trigger at `00:05 UTC` every single day (5 minutes after Groq resets the daily quotas).
- Newly labeled reviews are appended to `data/augmented/yelp_labeled_5000.jsonl`.
- Key quota states are tracked in `data/augmented/key_state.json`.
- When 5,000 items are reached, the pipeline marks itself complete!

---

## 6. Local / Alternative Usage

If you ever want to run the same script on your local machine or a cloud VPS:

```bash
# Run a single daily batch locally:
python scripts/run_daily_labeling.py --target-count 5000

# Run in continuous background daemon mode (for an always-on VPS):
python scripts/run_daily_labeling.py --daemon
```
