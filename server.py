import threading
import webbrowser
import os
import http.server
import socketserver

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Работаем в папке со скриптом

def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{PORT}/dash.html")

# Запускаем сервер
handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), handler)

print(f"🚀 Сервер запущен! Открываю браузер: http://127.0.0.1:{PORT}/dash.html")
print("Нажмите Ctrl+C, чтобы остановить сервер.")

# Открываем браузер в отдельном потоке через полсекунды, 
# чтобы сервер успел подняться
threading.Timer(1.0, open_browser).start()

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nСервер остановлен.")
    httpd.server_close()
