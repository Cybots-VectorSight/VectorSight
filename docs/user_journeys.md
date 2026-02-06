# VectorSight User Journeys

**How users interact with each app, what they input, what they expect.**

---

## Table of Contents

1. [User Types](#1-user-types)
2. [App 1: Analyzer & Chatbot](#2-app-1-analyzer--chatbot)
3. [App 2: Modifier](#3-app-2-modifier)
4. [App 3: Icon Set Expander](#4-app-3-icon-set-expander)
5. [App 4: Icon Creator](#5-app-4-icon-creator)
6. [App 5: SVG Playground](#6-app-5-svg-playground)
7. [Input Types Reference](#7-input-types-reference)
8. [Error States](#8-error-states)

---

## 1. User Types

### 1.1 Primary Users

| User Type | Description | Primary Goal |
|---|---|---|
| **Designer** | Creates icons, UI elements | Understand and modify SVGs |
| **Developer** | Builds products with icons | Get clean code, understand structure |
| **Hackathon Judge** | Evaluates the project | See the demo work, understand the innovation |

### 1.2 User Skill Levels

| Level | SVG Knowledge | Expectation |
|---|---|---|
| **Beginner** | None | Plain English, visual feedback |
| **Intermediate** | Basic | Some technical terms OK |
| **Expert** | Full | Wants raw data access |

**Design for beginners. Provide expert options.**

---

## 2. App 1: Analyzer & Chatbot

**Purpose:** Upload an SVG, see it analyzed, ask questions.

### 2.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  [Upload SVG]  [Paste Code]  [Use Sample]               │
├───────────────────────────┬─────────────────────────────┤
│                           │                             │
│   SVG Preview             │   Chatbot                   │
│   with overlays           │                             │
│                           │   [User message...]         │
│   [Toggle] Bounds         │                             │
│   [Toggle] Grid           │   AI: "This icon has..."    │
│   [Toggle] Symmetry       │                             │
│   [Toggle] Gaps           │                             │
│                           │                             │
├───────────────────────────┴─────────────────────────────┤
│  [Raw] [Shapes] [Grid] [Analysis] [Description] [Clean] │
│                                                         │
│  Tab content shows selected column data                 │
└─────────────────────────────────────────────────────────┘
```

### 2.2 User Journey

| Step | User Action | System Response |
|---|---|---|
| 1 | Arrives at page | See empty state with upload prompt |
| 2 | Uploads SVG (drag, click, or paste) | Preview appears, processing starts |
| 3 | Sees loading state | "Analyzing spatial structure..." |
| 4 | Analysis complete | Overlays appear, tabs populate, chat ready |
| 5 | Toggles overlays | Visual feedback on SVG preview |
| 6 | Asks question in chat | AI responds with spatial context |
| 7 | Clicks tabs | Sees raw data for each column |

### 2.3 Input Methods

| Method | How it works |
|---|---|
| **Drag & drop** | Drop .svg file onto upload zone |
| **Click to upload** | File picker, .svg filter |
| **Paste code** | Text area for raw SVG code |
| **Use sample** | Dropdown of test icons (smiley, home, etc.) |
| **URL** | Paste URL to SVG file (stretch goal) |

### 2.4 Chat Input Types

Users will ask questions. Here are the categories:

#### Measurement questions
```
"How far apart are the eyes?"
"What's the stroke width?"
"How big is the outer circle?"
"What percentage of the canvas is filled?"
```

#### Relationship questions
```
"Is the door inside the house?"
"What's above the mouth?"
"Are the eyes aligned?"
"Is this symmetric?"
```

#### Description questions
```
"What does this look like?"
"Describe this icon"
"What category is this?"
"What are the main elements?"
```

#### Comparison questions
```
"Which shape is bigger?"
"Is the left eye the same size as the right?"
"What's the largest element?"
```

#### Modification questions (leads to App 2)
```
"How would I make this thicker?"
"What if I moved the eyes closer?"
"Can you make it wink?"
```

### 2.5 Sample Responses

**User:** "How far apart are the eyes?"

**AI Response:**
```
The eyes are 5 units apart (edge to edge).

Details:
- Left eye: circle at (8, 9), radius 1.5
- Right eye: circle at (16, 9), radius 1.5
- Center-to-center distance: 8 units
- Edge-to-edge gap: 5 units (8 - 1.5 - 1.5)
- As percentage of icon width: 20.8%
```

**User:** "Is this symmetric?"

**AI Response:**
```
Yes, this icon is vertically symmetric.

- Left eye at x=8, right eye at x=16 (both 4 units from center)
- Both eyes same size (radius 1.5)
- Mouth centered at x=12
- Face outline centered at x=12

No horizontal symmetry (smile only curves down).
```

---

## 3. App 2: Modifier

**Purpose:** Describe changes in English, see them applied.

### 3.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  [Back to Analyzer]                                     │
├─────────────────────────┬───────────────────────────────┤
│                         │                               │
│   BEFORE                │   AFTER                       │
│   Original SVG          │   Modified SVG                │
│                         │                               │
├─────────────────────────┴───────────────────────────────┤
│                                                         │
│  Describe your change:                                  │
│  [Make the strokes thicker_________________________]    │
│                                                         │
│  [Apply Change]                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  History: [Change 1] [Change 2] [Change 3] [Undo All]   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 User Journey

| Step | User Action | System Response |
|---|---|---|
| 1 | Comes from Analyzer (or uploads new) | Original SVG in "Before" pane |
| 2 | Types modification request | - |
| 3 | Clicks "Apply Change" | Loading state |
| 4 | Modification complete | "After" pane shows result |
| 5 | Accepts or requests another change | History updates |
| 6 | Can undo any change | Reverts to previous state |

### 3.3 Modification Input Types

#### Style changes
```
"Make the strokes thicker"
"Change stroke width to 3"
"Make it bolder"
"Use thinner lines"
"Add rounded corners"
```

#### Color changes
```
"Make it red"
"Change the fill to blue"
"Make the background transparent"
"Use a gradient"
```

#### Position changes
```
"Move the eyes closer together"
"Center the door"
"Shift everything left"
"Make the roof taller"
```

#### Addition
```
"Add a hat on top"
"Put a dot in the center"
"Add eyebrows"
"Include a chimney"
```

#### Removal
```
"Remove the smile"
"Delete the right eye"
"Take away the door"
```

#### Transformation
```
"Make it wink"
"Turn the smile into a frown"
"Rotate 45 degrees"
"Flip horizontally"
"Make it 2x bigger"
```

### 3.4 Sample Interaction

**User:** "Make the eyes closer together"

**AI Response:**
```
Done! I moved the eyes 2 units closer to center.

Before:
- Left eye: x = 8
- Right eye: x = 16
- Gap: 5 units

After:
- Left eye: x = 10
- Right eye: x = 14
- Gap: 1 unit

The eyes are now 80% closer. Want me to adjust further?
```

---

## 4. App 3: Icon Set Expander

**Purpose:** Upload a set of icons, extract rules, generate matching icons.

### 4.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  Upload Icon Set                                        │
│  [Drop multiple SVGs here or click to upload]           │
├─────────────────────────────────────────────────────────┤
│  Uploaded (5):  [home] [user] [mail] [settings] [+2]    │
├─────────────────────────────────────────────────────────┤
│  Extracted Rules:                                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │ • Stroke width: 2px                             │    │
│  │ • Canvas: 24x24                                 │    │
│  │ • Padding: 12% (3 units each side)              │    │
│  │ • Style: Outline only, no fills                 │    │
│  │ • Corners: Rounded (radius 1)                   │    │
│  │ • Line caps: Round                              │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  Generate new icon:                                     │
│  [Shopping cart icon in the same style______________]   │
│  [Generate]                                             │
├─────────────────────────────────────────────────────────┤
│  Generated:  [Preview]  [Download SVG]  [Add to Set]    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 User Journey

| Step | User Action | System Response |
|---|---|---|
| 1 | Uploads multiple SVGs | Icons appear in grid |
| 2 | Waits for analysis | "Extracting design rules..." |
| 3 | Sees extracted rules | Rules panel populates |
| 4 | Types description of new icon | - |
| 5 | Clicks Generate | Loading, then preview appears |
| 6 | Reviews generated icon | Can tweak or accept |
| 7 | Downloads or adds to set | SVG file or adds to collection |

### 4.3 Input Types

#### Icon set upload
```
Multiple .svg files (drag or select)
ZIP file containing SVGs
Folder selection (if supported)
```

#### New icon requests
```
"Shopping cart"
"Download arrow"
"Lock icon"
"Calendar"
"Notification bell with badge"
"Play button, triangular"
```

---

## 5. App 4: Icon Creator

**Purpose:** Describe a new icon from scratch, generate with spatial awareness.

### 5.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  Describe your icon:                                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ A simple house icon, outline style, 24x24       │    │
│  └─────────────────────────────────────────────────┘    │
│  [Generate]                                             │
├───────────────────────────┬─────────────────────────────┤
│                           │  Suggestions:               │
│   Generated Icon          │                             │
│   [Preview]               │  "Roof is 45% of height.    │
│                           │   Typical range: 30-40%.    │
│                           │   Adjust?"                  │
│                           │                             │
│                           │  [Make roof shorter]        │
│                           │  [Keep as is]               │
├───────────────────────────┴─────────────────────────────┤
│  [Download SVG]  [Send to Modifier]  [New Icon]         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 User Journey

| Step | User Action | System Response |
|---|---|---|
| 1 | Types icon description | - |
| 2 | Clicks Generate | Loading state |
| 3 | Sees generated icon | Preview with spatial suggestions |
| 4 | Reviews suggestions | AI notes about proportions, balance |
| 5 | Accepts or adjusts | Regenerates with tweaks |
| 6 | Downloads final | SVG file |

### 5.3 Input Types

#### Basic descriptions
```
"House"
"Heart"
"Star"
"Arrow pointing right"
```

#### Detailed descriptions
```
"A simple house icon, outline style, 24x24"
"Heart shape, filled, with rounded top curves"
"Five-pointed star, outline only, symmetric"
"Right arrow with a tail, not just a chevron"
```

#### Style specifications
```
"Minimalist"
"Bold strokes"
"Thin lines"
"Rounded corners"
"Sharp edges"
"Filled"
"Outline only"
```

#### Size specifications
```
"24x24"
"16x16"
"32x32"
"Same size as a Lucide icon"
```

---

## 6. App 5: SVG Playground

**Purpose:** Click any part of an icon, AI does something creative.

### 6.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  [Upload]  [Sample: Smiley]  [Sample: House]  [Sample]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                                                         │
│              [Large Interactive SVG]                    │
│                                                         │
│              Click any part!                            │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  Last action: "Clicked right eye → Made it wink"        │
│  [Undo]  [Reset]                                        │
└─────────────────────────────────────────────────────────┘
```

### 6.2 User Journey

| Step | User Action | System Response |
|---|---|---|
| 1 | Loads icon (upload or sample) | Large preview appears |
| 2 | Hovers over parts | Subtle highlight shows clickable regions |
| 3 | Clicks a part | Animation/change happens |
| 4 | Sees result | Icon updates, action logged |
| 5 | Clicks another part | Another creative response |
| 6 | Can undo or reset | Returns to previous state |

### 6.3 Interaction Map

#### Smiley face

| Click Target | AI Action |
|---|---|
| Right eye | Winks (circle → curved line) |
| Left eye | Winks |
| Mouth | Cycles: smile → frown → O → tongue |
| Face outline | Wobbles (animation) |
| Empty space inside | Adds blush marks |

#### House

| Click Target | AI Action |
|---|---|
| Door | Opens (rectangle splits) |
| Window | Light turns on (fill changes) |
| Chimney | Smoke appears (animated path) |
| Roof | Changes color / adds snow |

#### Settings gear

| Click Target | AI Action |
|---|---|
| Center circle | Pulses (animation) |
| Gear teeth | Rotates (animation) |
| Any tooth | Changes tooth count |

### 6.4 Input Types

**This app has minimal text input.** Interaction is mostly clicks.

| Input | Method |
|---|---|
| Load icon | Upload, sample buttons |
| Interact | Click on SVG parts |
| Undo | Button |
| Reset | Button |

---

## 7. Input Types Reference

### 7.1 File Inputs

| Type | Accepted | Validation |
|---|---|---|
| SVG file | `.svg` | Valid XML, has `<svg>` root |
| Multiple SVGs | `.svg` × n | Same as above |
| Pasted code | Text | Valid SVG markup |

### 7.2 Text Inputs

| Category | Examples |
|---|---|
| Questions | "How far...", "What is...", "Is this..." |
| Commands | "Make it...", "Change the...", "Add a..." |
| Descriptions | "A house icon...", "Simple heart shape..." |

### 7.3 Click Inputs

| Target | Detection Method |
|---|---|
| Shape | SVG element hit testing |
| Region | Grid cell mapping |
| Empty space | Inverse of shapes |

---

## 8. Error States

### 8.1 Upload Errors

| Error | Message | Recovery |
|---|---|---|
| Not an SVG | "Please upload an SVG file" | Show accepted formats |
| Invalid SVG | "This SVG couldn't be parsed" | Show what went wrong |
| Too large | "SVG is too complex (>1MB)" | Suggest simplifying |
| Empty file | "This file appears to be empty" | Prompt re-upload |

### 8.2 Processing Errors

| Error | Message | Recovery |
|---|---|---|
| Parse failed | "Couldn't analyze this SVG" | Show original anyway |
| Timeout | "Analysis is taking too long" | Offer retry or skip |
| API error | "Couldn't generate description" | Show other columns |

### 8.3 Chat Errors

| Error | Message | Recovery |
|---|---|---|
| No SVG loaded | "Upload an SVG first to ask questions" | Highlight upload |
| Can't answer | "I couldn't determine that from the spatial data" | Suggest rephrasing |
| Rate limit | "Too many requests, please wait" | Show cooldown |

---

**Next:** See `data_spec.md` for input data details and output schemas.
