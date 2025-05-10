# SPDX-FileCopyrightText: 2021 Limor Fried/ladyada for Adafruit Industries
# SPDX-FileCopyrightText: 2021 Melissa LeBlanc-Williams for Adafruit Industries
#
# SPDX-License-Identifier: MIT

import time
import logging
import argparse

import cv2
import pygame
import os
import subprocess
import sys
import numpy as np
import signal

import board
import pwmio
import digitalio
from adafruit_motor import servo
import threading
from threading import Lock
import adafruit_dotstar
import sounddevice as sd
import json
import requests
import whisper
import tempfile

# Global constants
DOTSTAR_DATA = board.D5
DOTSTAR_CLOCK = board.D6
SERVO_PIN = board.D12
RECYCLE_CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic"]
INFERENCE_INTERVAL = 0.2
CONFIDENCE_THRESHOLD = 0.8
PERSISTANCE_THRESHOLD = 0.75

# Initialization
action_lock = Lock()
dots = adafruit_dotstar.DotStar(DOTSTAR_CLOCK, DOTSTAR_DATA, 3, brightness=0.2)
n_dots = len(dots)
model = whisper.load_model("tiny.en")

# Helper functions
def call_ollama(prompt, model="llama3.2:1b", max_tokens=100):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return ""
    
def llm_generate_recycle_message(object_name):
    prompt = (
        f"You are a funny recycling bin. Respond with a short, humorous thank-you "
        f"for recycling {object_name}. One sentence only. \n\nResponse:"
    )
    return call_ollama(prompt, model="llama3.2:1b", max_tokens=40)

def llm_generate_recycle_question():
    prompt = (
        "You're a recycling bin that talks to users. An unknown item just landed inside you.\n"
        "You need to ask the user if the item is recyclable.\n"
        "Ask a clear, short, direct yes/no question. Do NOT guess what the item is.\n"
        "Avoid jokes, metaphors, or humor. Be polite and concise.\n"
        "Only ask one question. Keep it short. End with a question mark.\n"
        "Examples:\n"
        "- \"Do you think this item is recyclable?\"\n"
        "- \"Is this something that should go in the recycling bin?\"\n"
        "- \"Is this item recyclable, yes or no?\"\n"
        "- \"Would you say this belongs in the recycling bin?\"\n"
        "\nResponse:"
    )
    return call_ollama(prompt, model="llama3.2:1b", max_tokens=40)

def llm_judge_user_reply(user_input):
    prompt = (
        "You are a recycling bin. A person answered your question about whether something is recyclable.\n"
        f"User said: \"{user_input}\"\n\n"
        "Reply ONLY with this JSON format:\n"
        "{\"recyclable\": true/false, \"reply\": \"<short funny response>\"}\n\n"
        "Examples:\n"
        "- \"Yes\" → {\"recyclable\": true, \"reply\": \"Opening up! Here we go!\"}\n"
        "- \"No\" → {\"recyclable\": false, \"reply\": \"Fair enough, staying closed.\"}\n"
        "- \"Maybe?\" → {\"recyclable\": false, \"reply\": \"Hmm, I need a clear answer!\"}\n"
        "- \"It's trash.\" → {\"recyclable\": false, \"reply\": \"Got it, no recycling magic today.\"}\n"
        "- \"Definitely recyclable!\" → {\"recyclable\": true, \"reply\": \"Perfect! Let’s do this!\"}\n"
        "\nNow your turn:\n"
        f"User: \"{user_input}\"\n"
        "JSON:"
    )
    result = call_ollama(prompt, model="llama3.2:1b", max_tokens=40)
    try:
        return json.loads(result)
    except Exception:
        print("[LLM Error] Failed to parse JSON:", result)
        return {"recyclable": False, "reply": "Hmm... I didn't quite understand that. Let's skip it for now."}

def record_and_transcribe(duration=5):
    fs = 16000
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        from scipy.io.wavfile import write
        write(f.name, fs, audio)
        result = model.transcribe(f.name, fp16=False)  # fp16=False for CPU
        print("[Whisper]:", result["text"])
        return result["text"].strip()

def speak(text):
    tmp_file = "/tmp/speech.wav"
    try:
        safe_text = text.replace('"', '').replace("'", '')
        subprocess.run(['pico2wave', '--lang=en-US', '--wave=' + tmp_file, safe_text], check=True)
        subprocess.run(['aplay', tmp_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[TTS Error] Failed to generate or play audio: {e}")
    except Exception as e:
        print(f"[TTS Error] Unknown error: {e}")
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def set_dotstar_state(state):
    if state == 0:
        color = (0, 0, 0)
    elif state == 1:
        color = (0, 255, 0)  # green
    elif state == 2:
        color = (255, 0, 0)  # red
    else:
        print(f"Invalid state: {state}")
        return

    for dot in range(n_dots):
        dots[dot] = color

def release_servo_pin():
    pin = digitalio.DigitalInOut(SERVO_PIN)
    pin.direction = digitalio.Direction.OUTPUT
    pin.value = False

# def servo_action():
#     pwm = pwmio.PWMOut(SERVO_PIN, duty_cycle=2 ** 15, frequency=50)
#     my_servo = servo.Servo(pwm)
#     my_servo.angle = 180
#     set_dotstar_state(1)
#     time.sleep(2)
#     my_servo.angle = 0
#     set_dotstar_state(0)
#     time.sleep(0.2)
#     pwm.deinit()
#     release_servo_pin()

def recycle_action(text):
    action_lock.acquire()
    try:
        pwm = pwmio.PWMOut(SERVO_PIN, duty_cycle=2 ** 15, frequency=50)
        my_servo = servo.Servo(pwm)
        my_servo.angle = 180
        set_dotstar_state(1)
        speak(text)
        my_servo.angle = 0
        set_dotstar_state(0)
        time.sleep(0.2)
        pwm.deinit()
        release_servo_pin()
    finally:
        action_lock.release()

def trash_action():
    if not action_lock.acquire(blocking=False):
        return
    try:
        print("Starting trash interaction...")

        # Step 1: Ask user
        question = llm_generate_recycle_question()
        print("[LLM Question]:", question)
        set_dotstar_state(2)
        speak(question)

        # Step 2: Listen to user reply
        user_input = record_and_transcribe(duration=5)
        if not user_input:
            speak("Sorry, I didn't hear anything. Maybe try again later.")
            set_dotstar_state(0)
            return

        # Step 3: Feed to LLM and interpret
        result = llm_judge_user_reply(user_input)
        reply = result.get("reply", "Thanks for letting me know.")
        is_recyclable = result.get("recyclable", False)
        print("[LLM Reply]:", reply)
        print("[LLM Recyclable]:", is_recyclable)

        # Step 4: Respond + act
        if is_recyclable:
            threading.Thread(target=recycle_action, args=(reply,), daemon=True).start()
        else:
            speak(reply)
            set_dotstar_state(0)
    finally:
        action_lock.release()

def dont_quit(signal, frame):
   print('Caught signal: {}'.format(signal))
signal.signal(signal.SIGHUP, dont_quit)

def get_output_probs(interpreter, output_details):
    """Get dequantized output probabilities from a TFLite interpreter."""
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_dtype = output_details[0]['dtype']

    if output_dtype == np.float32:
        return raw_output.astype(np.float32)

    # Handle quantized output (uint8 / int8)
    quant_params = output_details[0]['quantization_parameters']
    scales = quant_params['scales']
    zero_points = quant_params['zero_points']

    if len(scales) == 0 or len(zero_points) == 0:
        raise ValueError("Quantized model missing scale or zero_point info.")

    scale = scales[0]
    zero_point = zero_points[0]

    return scale * (raw_output.astype(np.float32) - zero_point)

# App
from rpi_vision.agent.capturev2 import PiCameraStream

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
capture_manager = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')
    
    parser.add_argument('--model-path', type=str,
                        dest='model_path', default=None,
                        help='(optional) Path to a custom TFLite model')
    
    parser.add_argument('--labels', type=str, default=None,
                    help='Path to label map text file (e.g. labels.txt)')

    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270],
                        dest='rotation', action='store', default=0,
                        help='Rotate everything on the display by this amount')
    args = parser.parse_args()
    return args

last_seen = [None] * 10
last_spoken = None

def main(args):
    global last_spoken, capture_manager, last_seen

    capture_manager = PiCameraStream(preview=False)

    if args.rotation in (0, 180):
        buffer = pygame.Surface((screen.get_width(), screen.get_height()))
    else:
        buffer = pygame.Surface((screen.get_height(), screen.get_width()))

    pygame.mouse.set_visible(False)
    screen.fill((0,0,0))
    try:
        splash = pygame.image.load(os.path.join(os.path.dirname(sys.argv[0]), 'bchatsplash.bmp'))
        splash = pygame.transform.rotate(splash, args.rotation)
        # Scale the square image up to the smaller of the width or height
        splash = pygame.transform.scale(splash, (min(screen.get_width(), screen.get_height()), min(screen.get_width(), screen.get_height())))
        # Center the image
        screen.blit(splash, ((screen.get_width() - splash.get_width()) // 2, (screen.get_height() - splash.get_height()) // 2))

    except pygame.error:
        pass
    pygame.display.update()

    # Let's figure out the scale size first for non-square images
    scale = max(buffer.get_height() // capture_manager.resolution[1], 1)
    scaled_resolution = tuple([x * scale for x in capture_manager.resolution])

    # use the default font, but scale it
    smallfont = pygame.font.Font(None, 24 * scale)
    medfont = pygame.font.Font(None, 36 * scale)
    bigfont = pygame.font.Font(None, 48 * scale)

    if args.model_path:
        if not args.labels:
            print("[ERROR] --labels is required when using --model-path.")
            print("Example: python3 main.py --model-path=model.tflite --labels=labels.txt")
            sys.exit(1)
        elif not os.path.exists(args.labels):
            print(f"[ERROR] Label file '{args.labels}' not found.")
            sys.exit(1)

    label_map = []
    if args.labels:
        with open(args.labels, 'r') as f:
            label_map = [line.strip() for line in f.readlines()]

    if args.model_path:
        # custom tflite
        from tensorflow.lite.python.interpreter import Interpreter
        interpreter = Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

    else:
        # default mobilenetv2 model
        from rpi_vision.models.mobilenet_v2 import MobileNetV2Base
        model = MobileNetV2Base(include_top=args.include_top)

    capture_manager.start()

    last_inference = time.monotonic() - INFERENCE_INTERVAL

    detecttext_surface = None
    detecttext_position = None

    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue
        buffer.fill((0,0,0))
        frame = capture_manager.read()
        # get the raw data frame & swap red & blue channels
        previewframe = np.ascontiguousarray(capture_manager.frame)
        # make it an image
        img = pygame.image.frombuffer(previewframe, capture_manager.resolution, 'RGB')
        img = pygame.transform.scale(img, scaled_resolution)

        cropped_region = (
            (img.get_width() - buffer.get_width()) // 2,
            (img.get_height() - buffer.get_height()) // 2,
            buffer.get_width(),
            buffer.get_height()
        )

        # draw it!
        buffer.blit(img, (0, 0), cropped_region)

        if action_lock.locked():
            buffer.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))
            screen.blit(pygame.transform.rotate(buffer, args.rotation), (0,0))
            pygame.display.update()
            time.sleep(0.1)
            continue

        timestamp = time.monotonic()
        if timestamp - last_inference < INFERENCE_INTERVAL:
            continue
        last_inference = timestamp

        if args.model_path:
            # custom tflite
            resized_frame = cv2.resize(frame, (width, height))
            if channels == 1:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)  # RGB → Gray
                resized_frame = np.expand_dims(resized_frame, axis=-1)           # Add channel dim
                
            # Normalize or quantize input
            if input_dtype == np.float32:
                input_data = resized_frame.astype(np.float32) / 255.0
            else:
                input_data = resized_frame.astype(input_dtype)

            # input_data = np.array(resized_frame, dtype=input_dtype)
            input_data = np.expand_dims(input_data, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_probs = get_output_probs(interpreter, output_details)

            top_index = int(np.argmax(output_probs))
            confidence = float(output_probs[top_index])
            prediction = [(top_index, label_map[top_index], confidence)] if confidence > CONFIDENCE_THRESHOLD else []
        else:
            if args.tflite:
                prediction = model.tflite_predict(frame)[0]
            else:
                prediction = model.predict(frame)[0]
        
        logging.info(prediction)
        delta = time.monotonic() - timestamp
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if args.tflite else "TF", delta * 1000, 1 / delta))
        print(last_seen)

        # add FPS & temp on top corner of image
        # fpstext = "%0.1f FPS" % (1/delta,)
        # fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        # fpstext_position = (buffer.get_width()-10, 10) # near the top right corner
        # buffer.blit(fpstext_surface, fpstext_surface.get_rect(topright=fpstext_position))
        # try:
        #     temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
        #     temptext = "%d\N{DEGREE SIGN}C" % temp
        #     temptext_surface = smallfont.render(temptext, True, (255, 0, 0))
        #     temptext_position = (buffer.get_width()-10, 30) # near the top right corner
        #     buffer.blit(temptext_surface, temptext_surface.get_rect(topright=temptext_position))
        # except OSError:
        #     pass

        for p in prediction:
            label, name, conf = p
            if conf > CONFIDENCE_THRESHOLD:
                print("Detected", name)

                persistant_obj = False  # assume the object is not persistant
                last_seen.append(name)
                last_seen.pop(0)

                inferred_times = last_seen.count(name)
                if inferred_times / len(last_seen) > PERSISTANCE_THRESHOLD:  # over quarter time
                    persistant_obj = True

                detecttext = name.replace("_", " ")
                detecttextfont = None
                for f in (bigfont, medfont, smallfont):
                    detectsize = f.size(detecttext)
                    if detectsize[0] < screen.get_width(): # it'll fit!
                        detecttextfont = f
                        break
                else:
                    detecttextfont = smallfont # well, we'll do our best
                
                #detecttext_color = (0, 255, 0) if persistant_obj else (255, 255, 255)
                if not persistant_obj:
                    detecttext_color = (255, 255, 255)
                else:
                    detecttext_color = (0, 255, 0) if name in RECYCLE_CATEGORIES else (255, 0, 0)
                detecttext_surface = detecttextfont.render(detecttext, True, detecttext_color)
                detecttext_position = (buffer.get_width()//2,
                                       buffer.get_height() - detecttextfont.size(detecttext)[1])
                buffer.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))

                # if persistant_obj and last_spoken != detecttext:
                #     subprocess.call(f"echo {detecttext} | festival --tts &", shell=True)
                #     last_spoken = detecttext

                if last_seen.count(name) == len(last_seen) and not action_lock.locked():
                    last_seen = [None] * 10
                    if name in RECYCLE_CATEGORIES:
                        message = llm_generate_recycle_message(name)
                        print("[LLM Message]:", message)
                        # speak(message)
                        # print("Activating motor")
                        threading.Thread(target=recycle_action, args=(message,), daemon=True).start()
                    else:
                        threading.Thread(target=trash_action, daemon=True).start()

                break
        else:
            last_seen.append(None)
            last_seen.pop(0)
            if last_seen.count(None) == len(last_seen):
                last_spoken = None

        screen.blit(pygame.transform.rotate(buffer, args.rotation), (0,0))
        pygame.display.update()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
