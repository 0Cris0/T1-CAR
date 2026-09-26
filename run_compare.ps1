<#
.SYNOPSIS
    sweep.ps1 -- corre bs_numpy.py con todas las combinaciones (p, t) tales
                 que p * t <= p_max, y vuelca la salida a <machine_name>.log

.USAGE
    .\sweep.ps1 -PMax 16 -ExtraArgs "--n-resamples","2000","--repeats","3"

.EJEMPLO
    .\sweep.ps1 -PMax 16
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [int]$PMax,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$Script      = "bs_numpy.py"
$MachineName = $env:COMPUTERNAME
$LogFile     = "$MachineName.log"

function Write-Log {
    param([string]$Message)
    $Message | Tee-Object -FilePath $LogFile -Append
}

Write-Log "Sweep iniciado $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') -- host=$MachineName p_max=$PMax"

for ($p = 1; $p -le $PMax; $p++) {
    $tMax = [math]::Floor($PMax / $p)
    for ($t = 1; $t -le $tMax; $t++) {

        Write-Log ""
        Write-Log "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') p=$p inner-threads=$t (p*t=$($p*$t)) ==="

        # Se captura tanto stdout como stderr y se anexan al log,
        # y ademas se muestran en pantalla en tiempo real.
        $pyArgs = @("$Script", "-p", "$p", "--inner-threads", "$t") + $ExtraArgs
        & python @pyArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    }
}

Write-Log ""
Write-Log "Sweep finalizado $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"