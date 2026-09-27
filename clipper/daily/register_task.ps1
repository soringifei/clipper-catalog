[CmdletBinding()]
param([string]$At='05:30')
# Registers the hidden "HookHaus Daily" job: every day at 05:30 pythonw runs make_daily.py for that day.
# pythonw = no console window, no focus; ffmpeg/whisper children start with CREATE_NO_WINDOW at below-normal priority.
# If a game is running it waits up to 90 min, then renders with light CPU settings (libx264 veryfast, 4 threads).
# The job only prepares clips and a board approval task; it never publishes or schedules anything on TikTok.
# Pause: Disable-ScheduledTask 'HookHaus Daily'   Resume: Enable-ScheduledTask 'HookHaus Daily'
$ErrorActionPreference='Stop'
$daily=$PSScriptRoot
$pythonw=Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\pythonw.exe'
$script=Join-Path $daily 'make_daily.py'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal=New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$settings.WakeToRun=$false
$settings.RestartCount=0
$settings.Priority=7
$trigger=New-ScheduledTaskTrigger -Daily -At $At
$action=New-ScheduledTaskAction -Execute $pythonw -Argument ('"'+$script+'" --wait-for-game 90') -WorkingDirectory $daily
Register-ScheduledTask -TaskName 'HookHaus Daily' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description ('Clipper owned: '+$daily+'; hidden HookHaus daily batch (7 clips from rights-cleared sources, review.html + board approval task). Prepares only, never publishes.') -Force | Out-Null
Get-ScheduledTask -TaskName 'HookHaus Daily' | Select-Object TaskName,State,@{n='Next';e={(Get-ScheduledTaskInfo $_).NextRunTime}}
