# Allxon_AIBox_Backend
### Setup
Jetson 新機軟體安裝請跟著[官方文件](https://docs.nvidia.com/jetson/jps/setup/quick-start.html)進行安裝，並且在 **Start Application** 章節有將服務成功開啟。

安裝相關套件：
```
pip3 install -r requirements.txt
```

系統監控相關：
node_exporter
```
docker run -d \
  --net="host" \
  --pid="host" \
  -v "/:/host:ro,rslave" \
  quay.io/prometheus/node-exporter:latest \
  --path.rootfs=/host
```

prometheus
```
docker run --name prometheus -d -p 127.0.0.1:9090:9090 prom/prometheus
```

安裝完畢後，開瀏覽器到 `http://127.0.0.1:9090/targets`，查看兩個服務的 status 是否為 up（在畫面右邊）
若無請在群組通知，會再協助除錯，感謝。

### Qcickstart

開啟 containers：
```
jetson-containers run --workdir /opt/nanoowl --volume $(pwd):/app $(autotag nanoowl)
```
```
sudo docker start node-exporter
```
```
sudo docker start prometheus
```

於容器內裝套件 （先在容器裡裝，之後會整到 dockerfile 裡）& 執行 ai app 後端：
```
apt-get update && apt-get install -y ffmpeg && pip3 install ffmpeg-python && pip install apscheduler && ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime
```

於容器內開啟後端：
```
jetson-containers run --name jetson_container_20250118_231824 --workdir /opt/nanoowl --volume $(pwd):/app $(autotag nanoowl)
```

於本機執行 camera manager：（用於抓取相機清單及其它輔助功能如抓取 GPU status 等）
```
python3 camara_manager.py
```

開啟前端後（請參考前端文件），進入 `127.0.0.1:8000`，可看見 AI APP。

### AI APP 說明：
 1. 開啟左側 *Video Stream* 後，按 + 選擇相機後，開始串流。
 2. *Audio Stream* 為開關串流聲音，預設為關閉（推薦關閉，尤其推論時）
 3.  *Video Infer.* 為開關推論，預設為關閉，若開啟才會進行推論。（可於 log 確認功能正常與否）
 4. *Select Area* 若 app 中有畫面，可選擇不規則四邊形的範圍來進行推論。
 5. *Prompt Panel* 輸入 prompt 使用，輸入後且開啟 *Video Infer* 才會進行推論。
 6. *Stream log* 與 *Settings* 目前無功能。

### logs 定期匯出
使用 cron 做定期匯出，需事前設定，步驟如下：
1. 編輯 cron 工作 `crontab -e`
2. 在文件內新增 `*/3 * * * * sudo docker cp jetson_container_20250118_231824:/opt/nanoowl/logs ~/Downloads/logs/`，儲存退出