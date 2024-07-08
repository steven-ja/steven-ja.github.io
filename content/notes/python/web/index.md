---
title: WebSocket
weight: 10
menu:
  notes:
    name: WebSocket
    identifier: notes-python-web
    parent: notes-python
    weight: 10
---

<!-- A Sample Program -->
{{< note title="Connect to a Websocket">}}
A sample **python** program is shown here.
  
```python
import websocket


def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(f"Encountered error: {error}")
    
def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")
    ws.send("Hello, Worldy!")

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:xxxx", # insert here you websocket addres
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
```

Run the program as below:

```bash
$ python websocket_example.py
```
{{< /note >}}

<!-- Declaring Variables

{{< note title="Variables" >}}
**Normal Declaration:**
```go
var msg string
msg = "Hello"
```

---

**Shortcut:**
```go
msg := "Hello"
```
{{< /note >}}


<!-- Declaring Constants -->

<!-- {{< note title="Constants" >}}
```go
const Phi = 1.618
```
{{< /note >}} -->