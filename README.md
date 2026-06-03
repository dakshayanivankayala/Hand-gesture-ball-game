# AI Hand-Controlled Ball Dodge Game

An interactive computer vision game built using Python, OpenCV, and MediaPipe. The game uses real-time hand tracking through a webcam, allowing players to control an on-screen character using finger movements.

## Features

* Real-time hand gesture control
* Split-screen webcam and gameplay interface
* Normal balls (+1 point)
* Gold balls (+5 points)
* Danger balls (Game Over)
* Dynamic difficulty scaling
* Score and timer tracking
* Restart and game-over functionality

## Technologies

* Python
* OpenCV
* MediaPipe
* NumPy
* Computer Vision

## How It Works

The webcam detects the player's hand and tracks the index finger position. The player's character follows the detected finger movement, allowing them to collect reward balls while avoiding danger balls. The game speed increases as the score grows, making gameplay progressively more challenging.

## Controls

* Move your index finger to control the player
* Press R to restart
* Press Q to quit
