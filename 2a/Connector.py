import socket
import threading
import time

class CoppeliaClient:
    def __init__(self, host='127.0.0.1', port=50002):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.recv_thread = None
        
        # Line sensors (5 sensors) - matches C struct
        self.line_sensors = [0.0] * 5  # left_corner, left, middle, right, right_corner
        
        # Proximity sensor
        self.proximity_distance = 1.0  # Default max distance
        
        # Color sensor (RGB values) - raw values 0.0-1.0
        self.color_r = 0.0
        self.color_g = 0.0
        self.color_b = 0.0
        
        # Buffer for incoming data
        self.buffer = ""

    def connect(self):
        """Connect to the CoppeliaSim wrapper and start receive thread"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(1.0)  # Set timeout for blocking operations
            
            # Start the receive thread
            self.running = True
            self.recv_thread = threading.Thread(target=self._receive_loop)
            self.recv_thread.daemon = True
            self.recv_thread.start()
            
            print(f"✓ Connected to CoppeliaSim wrapper at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def set_motor(self, left_speed, right_speed):
        """Send motor speed command to the wrapper - matches C function"""
        try:
            if self.sock and self.running:
                cmd = f"L:{left_speed:.2f};R:{right_speed:.2f}\n"
                self.sock.sendall(cmd.encode())
                return True
            return False
        except Exception as e:
            print(f"❌ Error sending motor command: {e}")
            return False
    
    # Alias for compatibility with existing code
    def send_motor_command(self, left_speed, right_speed):
        """Alias for set_motor to maintain compatibility"""
        return self.set_motor(left_speed, right_speed)
    
    def pick_box(self):
        """Send pick box command - matches C function"""
        try:
            if self.sock and self.running:
                cmd = "PICK\n"
                self.sock.sendall(cmd.encode())
                return True
            return False
        except Exception as e:
            print(f"❌ Error sending pick command: {e}")
            return False
    
    def drop_box(self):
        """Send drop box command - matches C function"""
        try:
            if self.sock and self.running:
                cmd = "DROP\n"
                self.sock.sendall(cmd.encode())
                return True
            return False
        except Exception as e:
            print(f"❌ Error sending drop command: {e}")
            return False
    
    # Blocking versions for compatibility
    def pick_box_blocking(self, timeout=2.0):
        """Pick box with blocking wait for response"""
        return self.pick_box()  # For now, just send command
    
    def drop_box_blocking(self, timeout=2.0):
        """Drop box with blocking wait for response"""
        return self.drop_box()  # For now, just send command
    
    def _receive_loop(self):
        """
        Thread function that continuously receives sensor data from the server
        Parses incoming data in format: S:val1,val2,val3,val4,val5;P:distance;C:r,g,b
        """
        line_buffer = ""
        
        while self.running:
            try:
                # Read data from socket
                data = self.sock.recv(1024)
                if not data:
                    break
                
                # Decode and add to buffer
                received = data.decode('utf-8', errors='ignore')
                line_buffer += received
                
                # Process complete lines
                while '\n' in line_buffer:
                    line, line_buffer = line_buffer.split('\n', 1)
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Parse the line: "S:val1,val2,val3,val4,val5;P:distance;C:r,g,b"
                    if line.startswith('S:'):
                        self._parse_sensor_line(line)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Error in receive loop: {e}")
                break
            
            time.sleep(0.001)  # Small delay to prevent excessive CPU usage
    
    def _parse_sensor_line(self, line):
        """Parse sensor data line - matches C parsing logic"""
        try:
            # Split by semicolons to get different sensor groups
            segments = line.split(';')
            
            for segment in segments:
                if segment.startswith('S:'):
                    # Line sensor data: "S:val1,val2,val3,val4,val5"
                    values_str = segment[2:]
                    values = values_str.split(',')
                    
                    # Update line sensors array (max 5 values)
                    for i, val_str in enumerate(values[:5]):
                        try:
                            self.line_sensors[i] = float(val_str)
                        except (ValueError, IndexError):
                            pass
                
                elif segment.startswith('P:'):
                    # Proximity sensor: "P:distance"
                    try:
                        self.proximity_distance = float(segment[2:])
                    except ValueError:
                        pass
                
                elif segment.startswith('C:'):
                    # Color sensor: "C:r,g,b"
                    values_str = segment[2:]
                    values = values_str.split(',')
                    
                    try:
                        if len(values) >= 1:
                            self.color_r = float(values[0])
                        if len(values) >= 2:
                            self.color_g = float(values[1])
                        if len(values) >= 3:
                            self.color_b = float(values[2])
                    except ValueError:
                        pass
        
        except Exception as e:
            print(f"❌ Error parsing sensor line: {e}")
    
    def get_sensors_blocking(self, timeout=1.0):
        """Get current sensor data - returns dictionary format for compatibility"""
        if not self.running:
            return None
        
        # Return current sensor values in dictionary format
        return {
            'line': {
                'left_corner': self.line_sensors[0],
                'left': self.line_sensors[1],
                'middle': self.line_sensors[2],
                'right': self.line_sensors[3],
                'right_corner': self.line_sensors[4]
            },
            'proximity': self.proximity_distance,
            'color': {
                'r': int(self.color_r * 255),  # Convert to 0-255 range for compatibility
                'g': int(self.color_g * 255),
                'b': int(self.color_b * 255)
            }
        }
    
    def get_line_sensors(self):
        """Get raw line sensor array - matches C interface"""
        return self.line_sensors.copy()
    
    def get_proximity_distance(self):
        """Get proximity distance - matches C interface"""
        return self.proximity_distance
    
    def get_color_rgb(self):
        """Get color RGB values - matches C interface"""
        return (self.color_r, self.color_g, self.color_b)
    
    def disconnect(self):
        """Cleanly disconnect and cleanup - matches C function"""
        self.running = False
        
        # Wait for receive thread to finish
        if self.recv_thread and self.recv_thread.is_alive():
            self.recv_thread.join(timeout=1.0)
        
        # Close socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        print("🔌 Connection closed")
    
    def close(self):
        """Alias for disconnect"""
        self.disconnect()
    
    # Legacy methods for compatibility
    def start_simulation(self):
        """Legacy method - not used with current wrapper"""
        pass

    def stop_simulation(self):
        """Legacy method - not used with current wrapper"""
        pass

    def request_sensor_data(self):
        """Legacy method - data is received automatically"""
        pass

    def receive_sensor_data(self):
        """Legacy method - use get_sensors_blocking instead"""
        return self.get_sensors_blocking()