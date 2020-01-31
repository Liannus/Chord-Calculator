# Chord-Calculator
COMP4102 Final project for determining guitar chords through computer vision.

## Table of Contents
- [Contributors](#contributors)
- [Summary](#summary)
- [Background](#background)
- [The Challenge](#the-challenge)
- [Goals and Deliverables](#goals-and-deliverables)
- [Schedule](#schedule)


## Contributors
- Harry Ismayilov
  + [github ]()
- Josh Gorman
  + [github](https://github.com/Liannus)
  + [website](https://joshgorman.ca/)
- Kushal Choksi
  + [github]()

## Summary
Chord calculator is an computer vision application capable of estimating guitar chords through recognition of common hand positions and locations on the fretboard. This combination allows for the future development of educational software as well as sheet music generators.

## Background
Due to group members currently learning computer vision techniques for the first time, Chord Calculator is estimated to change in techniques and implementation.

Using edge detection algorithms to determine locations of fingers on the fret board or machine learning algorithms to recognize common chord shapes are techniques currently being researched for implementation. Some papers containing details on edge detection and similar methods can be found below:

- Burns, Anne-Marie. “Computer Vision Methods for Guitarist Left-Hand Fingering Recognition.” DigiTool Stream Gateway Error, Input Deviees and Music Interaction Lab Schulich School of Music McGill University, Feb. 2007, [digitool.library.mcgill.ca/webclient/StreamGate?folder_id=0&amp;dvs=1580432298292~888](digitool.library.mcgill.ca/webclient/StreamGate?folder_id=0&amp;dvs=1580432298292~888).
- Kerdvibulvech, Chutisant, and Hideo Saito. “Real-Time Guitar Chord Recognition SystemUsing Stereo Cameras for SupportingGuitarists.” Hyper Vision Research Labratory, Department of Informationand Computer Science, Keio University, 28 Feb. 2007, [hvrl.ics.keio.ac.jp/paper/pdf/international_Journal/2007/chutisant_ECTI07.pdf](hvrl.ics.keio.ac.jp/paper/pdf/international_Journal/2007/chutisant_ECTI07.pdf).
- Scarr, Joseph, and Richard Green. “Retrieval of Guitarist Fingering Information UsingComputer Vision.” Joey Scarr, Department of Computer Science and Software EngineeringUniversity of Canterbury, New Zealand, [joey.scarr.co.nz/pdf/autotab.pdf](joey.scarr.co.nz/pdf/autotab.pdf).

## The Challenge
This problem present many issues that similar algorithms with other instruments manage to avoid:
- The guitar moves and therefore the fretboard must be tracked
- Frets and commonly tracked parts such as fingertips are often occluded
- Certain frets are held down without fingertips to create "bar chords".

Therefore, this project will have to take multiple steps by first isolating the fretboard, then attempting to determine points of interest and compare versus existing datasets to make estimates on chord formations. It is believed that this project therefore has the means to teach basics of both computer vision and machine learning.

## Goals and Deliverables
Due to the difficulty of estimating software projects combined with the inexperience of group members, the minimum viable product will be **minimal** to best estimate completion.

The base level of success sought throughout this project will be the recognition of simple (non-bar) chords with a stationary guitar.

### Plan to Achieve
- Recognize and box off the fretboard
- Determine fingertip locations
- Estimate chord from known fingertip location

### Hope to Achieve
- Output chord location to sheet music
- Handle bar chords by comparing to images of common chord shapes
- Quick enough algorithm to recognize chords in succession during normal play.

## Schedule
#### feb 1  - feb 7
research open-cv functions and possible algorithms for implementation

#### feb 8  - feb 14

#### feb 15 - feb 21

#### feb 22 - feb 28

#### feb 29 - mar 6

#### mar 7  - mar 13

#### mar 14 - mar 20

#### mar 21 - mar 27

#### mar 28 - apr 3

#### mar 4  - apr 10




