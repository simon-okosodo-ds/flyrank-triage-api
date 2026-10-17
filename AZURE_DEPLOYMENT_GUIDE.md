# ☁️ Azure Free Deployment Guide for Flyrank Triage API

This guide provides a step-by-step walkthrough to deploy your **Flyrank Triage API** to **Microsoft Azure Container Apps (ACA)** for **100% free**, allowing you to showcase multi-cloud deployment (Render + Azure) on your CV and portfolio.

---

## 📌 Why Azure Container Apps (ACA)?

* 🆓 **100% Free Tier:** 180,000 vCPU-seconds + 2 Million HTTP requests **free every month forever**.
* ⚡ **Scale-to-Zero:** When idle, your container scales down to 0 instances — **consuming 0 billable resources**.
* 🐳 **Docker-Native:** Uses your existing `Dockerfile` without code changes.

---

## 🚀 Step 1: Create a Free Azure Account

Choose one of the two free account options:

1. **Azure for Students** *(Best if you have a `.edu` or university email)*:
   * URL: [azure.microsoft.com/free/students](https://azure.microsoft.com/free/students/)
   * Benefit: **$100 free annual credit + free services**.
   * **NO credit card required!**

2. **Azure Free Account** *(Standard)*:
   * URL: [azure.microsoft.com/free](https://azure.microsoft.com/free/)
   * Benefit: **$200 free credit (30 days)** + 12 months free popular services + Always Free tier.
   * *Requires a credit/debit card for identity verification ($0 charge).*

---

## 🛠️ Step 2: Choose Your Deployment Method

### Option A: Direct Deployment via Azure Portal (GUI - Easiest)

1. Sign in to the [Azure Portal](https://portal.azure.com).
2. Search for **Container Apps** in the top search bar and click **Create**.
3. **Basics Tab:**
   * **Subscription:** Select your Free / Student Subscription.
   * **Resource Group:** Click *Create new* -> enter `FlyrankResourceGroup`.
   * **Container App Name:** `flyrank-triage-api`
   * **Region:** Select `East US` (or closest region).
   * **Container Apps Environment:** Leave as default (creates new environment).
4. **App Source / Image Tab:**
   * Select **Quickstart image** or **GitHub Repository** / **Docker Hub**.
   * If using your Dockerfile from GitHub: Select **GitHub**, authenticate, select `flyrank-triage-api` repo, and set branch to `main`.
5. **Ingress Tab:**
   * **Ingress:** Enabled
   * **Traffic:** Accept traffic from anywhere (External).
   * **Target Port:** `10000` (matches the port exposed in `Dockerfile`).
6. Click **Review + Create**, then **Create**.

Your live API URL will look like:
`https://flyrank-triage-api.<random-string>.eastus.azurecontainerapps.io/docs`

---

### Option B: Command Line Deployment via Azure CLI

If you have [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed on your machine:

```bash
# 1. Log in to Azure
az login

# 2. Create a Resource Group
az group create --name FlyrankResourceGroup --location eastus

# 3. Deploy Container App directly from GitHub repo source
az containerapp up \
  --name flyrank-triage-api \
  --resource-group FlyrankResourceGroup \
  --location eastus \
  --repo https://github.com/simon-okosodo-ds/flyrank-triage-api \
  --target-port 10000 \
  --ingress external
```

---

## 🔄 Step 3: Enable Automated GitHub Actions CI/CD (Optional)

We have already created the GitHub Actions workflow file in `.github/workflows/deploy-azure.yml`.

To enable automatic deployment on every `git push`:

1. **Get Azure Service Principal Credentials:**
   Run in your terminal (using Azure CLI):
   ```bash
   az ad sp create-for-rbac --name "FlyrankGitHubCI" \
     --role contributor \
     --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/FlyrankResourceGroup \
     --json-auth
   ```
2. **Copy the JSON output**.
3. **Add Secret to GitHub:**
   * Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions**.
   * Click **New repository secret**.
   * **Name:** `AZURE_CREDENTIALS`
   * **Value:** Paste the complete JSON output from step 1.
4. Now, every push to `main` will automatically build your Docker container and deploy it to Azure Container Apps!

---

## 📋 How to Showcase Azure Deployment on Your CV / Resume

Add Azure alongside Render in your project description and skills section:

### Skills Section:
> **Cloud & Infrastructure:** Microsoft Azure (Container Apps), Render, Docker, FastAPI, CI/CD, GitHub Actions.

### Project Section:
> **Flyrank Triage API | Cloud & Machine Learning Project**
> * Deployed FastAPI ML triage engine across **Microsoft Azure Container Apps** and **Render**, leveraging Docker containers for multi-cloud deployment.
> * Configured scale-to-zero serverless container orchestration to maintain **100% free-tier resource compliance** while serving low-latency REST endpoints.
> * Implemented automated GitHub Actions CI/CD pipeline for container building, security checks, and continuous deployment.
