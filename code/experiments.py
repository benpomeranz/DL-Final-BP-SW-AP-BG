from assignment import train
from model_experimental import Recurrent
from preprocess import jsonl_to_data

def train_on_all():
    start_time = 1514782800
    end_time = 1546318800
    model = Recurrent()
    device_ids = ['002', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023'] #14,15,20,21 give NAN?
    for i in range(100):
        print(f"Epoch {i}")
        for device in device_ids:
            times, magnitudes, accels = jsonl_to_data(f"/Users/benpomeranz/Desktop/CS1470/DL-Final-BP-SW-AP-BG/big_data/device{device}_preprocessed", start_time, end_time)
            print(device)
            dist = train(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=True)[1]

if __name__ == "__main__":
    train_on_all()