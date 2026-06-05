<#
.SYNOPSIS
Provisions Azure Container Registry (ACR) and Azure Container Apps (ACA) for the Gov Schemes Voice Bot.

.DESCRIPTION
This script automates the creation of an Azure Resource Group, an Azure Container Registry,
builds the Docker image, pushes it to ACR, and provisions the Azure Container App with the required environment variables.

.PARAMETER ResourceGroup
Name of the Azure Resource Group to create/use.

.PARAMETER Location
Azure region for deployment.

.PARAMETER RegistryName
Name of the Azure Container Registry (must be globally unique and alphanumeric).

.PARAMETER AppEnvName
Name of the Azure Container Apps Environment.

.PARAMETER AppName
Name of the Azure Container App.
#>

param (
    [string]$ResourceGroup = "GovSchemesBotRG",
    [string]$Location = "centralindia",
    [string]$RegistryName = "govschemesacr$(Get-Random -Maximum 9999)",
    [string]$AppEnvName = "gov-schemes-env",
    [string]$AppName = "gov-schemes-bot"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting Azure Deployment Script..." -ForegroundColor Cyan

# 1. Create Resource Group
Write-Host "`n[1/5] Creating Resource Group: $ResourceGroup in $Location..."
az group create --name $ResourceGroup --location $Location | Out-Null

# 2. Create Azure Container Registry (Basic tier is sufficient for this)
Write-Host "`n[2/5] Creating Azure Container Registry: $RegistryName..."
az acr create --resource-group $ResourceGroup --name $RegistryName --sku Basic --admin-enabled true | Out-Null

# 3. Build and Push Docker Image using ACR Tasks (avoids needing local Docker daemon)
Write-Host "`n[3/5] Building and pushing Docker image to ACR (this may take a few minutes)..."
az acr build --registry $RegistryName --image "${AppName}:latest" .

# 4. Create Container Apps Environment
Write-Host "`n[4/5] Creating Container Apps Environment: $AppEnvName..."
az containerapp env create --name $AppEnvName --resource-group $ResourceGroup --location $Location | Out-Null

# 5. Create Container App
Write-Host "`n[5/5] Creating Container App: $AppName..."

# Get ACR credentials
$acrUsername = (az acr credential show --name $RegistryName --query "username" -o tsv)
$acrPassword = (az acr credential show --name $RegistryName --query "passwords[0].value" -o tsv)
$acrLoginServer = (az acr show --name $RegistryName --query "loginServer" -o tsv)

# Note: We create the app without secrets initially, but expose port 8000.
# The user will need to configure the actual .env secrets via GitHub Actions or the Azure Portal.
az containerapp create `
    --name $AppName `
    --resource-group $ResourceGroup `
    --environment $AppEnvName `
    --image "$acrLoginServer/${AppName}:latest" `
    --registry-server $acrLoginServer `
    --registry-username $acrUsername `
    --registry-password $acrPassword `
    --target-port 8000 `
    --ingress external `
    --min-replicas 0 `
    --max-replicas 5

Write-Host "`n✅ Deployment Complete!" -ForegroundColor Green
$appUrl = (az containerapp show --name $AppName --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv)
Write-Host "Your Container App is live at: https://$appUrl/webhook" -ForegroundColor Yellow
Write-Host "`nNext Steps:"
Write-Host "1. Go to the Azure Portal -> Container Apps -> $AppName -> Environment Variables."
Write-Host "2. Copy all variables from your local .env file into the Container App."
Write-Host "3. Update your Twilio WhatsApp Sandbox webhook with the URL above."
