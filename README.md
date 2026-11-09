# TypeScribe v1.0

Need to send out a hundred organic Christmas cards? Or you have a fifty page essay due and _Comic Sans_ doesn't quite cut it?
Got you covered.

**TypeScribe** is a tool that uses Machine Learning to convert typed text into realistic handwriting by introducing customizable degrees of randomness and variations, to make it all so believable.

<table align="center">
  <tr>
    <td><img src="assets/typescribe.png" alt="" width="300"></td>
    <td><img src="assets/1.png" alt="" width="300"></td>
    <td><img src="assets/2.png" alt="" width="300"></td>
    <td><img src="assets/3.png" alt="" width="300"></td>
  </tr>
      <tr>
    <td>TypeScribe</td>
    <td>Totally not homework</td>
    <td>A truly heartfelt letter</td>
    <td>Christmas Cards too!?</td>
  </tr>
</table>

### Features
- **Realistic Handwriting Generation with a Recurrent Neural Network (RNN)**
- **Choose from 12 predefined handwriting styles.**
- **Save complete settings as reusable profiles, and switch between them.**
- **Adjust line spacing, page size, margins, ink color, pen thickness, and more.**
- **User-friendly interface**
- **Automatically splits large texts into multiple lines, paragraphs and pages.**
- **Scalable SVG Output that maintains quality even on resizing**


### Getting Started

There are two ways to get things rolling:

1. **Download the Executable:**
   - Head to the [Releases page](https://github.com/rudyoactiv/typescribe-handwriting/releases) and download the **TypeScribe-v1.0.zip** package.
   - Unzip the file and double-click **TypeScribe.exe** to launch the application.

2. **Clone the Repository:**
   - Clone this repo using ```git clone https://github.com/rudyoactiv/typescribe-handwriting.git```
   - Navigate to the cloned directory.
   - With Anaconda installed, run ```conda env create -f environment.yml``` to create an environment.
   - Install the native GUI dependencies from your distro (GTK 4 + libadwaita + PyGObject), e.g.
     `sudo pacman -S python-gobject gtk4 libadwaita` (Arch), `sudo dnf install python3-gobject gtk4 libadwaita` (Fedora)
     or `sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1` (Debian/Ubuntu).
   - Launch the app with ```./gui.py``` (it automatically uses the system Python for the GUI and the `test_hand`
     conda environment for the model; set `TYPESCRIBE_PYTHON` to use a different interpreter for the model).
   - Optional: run ```./install-desktop.sh``` to add **TypeScribe** to your app launcher.

   The interface is a native GTK 4 / libadwaita app (GNOME style, adapts to light/dark mode). Generation runs in a
   background process, so the window stays responsive.


### Usage
1. Launch the application with the command ```./gui.py``` (or from your app launcher after `install-desktop.sh`), or double click TypeScribe.exe on Windows.
2. Enter your text and adjust settings via the interface.
3. Preview the layout and handwriting style.
4. Click **Generate**.
5. Wait while it generates your preview. May take a minute.
6. Click **Export** to export the generated pages. Select SVG and/or PNG in the
   export menu. Use **Export As** (`Ctrl+Shift+E`) to choose a destination;
   TypeScribe remembers that folder for later. `Ctrl+E` exports to the remembered
   folder.
7. To print the generated pages, choose **Print…** in the main menu (`Ctrl+P`).
   It uses the system print dialog, so "Print to File" works too. Whether the
   paper color and ruled lines are printed follows the page background setting in
   the Export dialog.

### Known Issues
- Antivirus programs may flag (and even delete) the executable. This is due to the way Python programs are built and handled by Windows and cannot possibly be fixed. However, there is no cause for concern.

### Future Plans
- A more responsive design
- More customization options
- Increasing the valid character set
- Fix antivirus false positives
- Build an installer to replace the zip
- Extra text formatting options (vary alignment, ink colors in a single document).

### Acknowledgements
This project is inspired by the work in the [handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis) repository by **sjvasquez**, which provides the foundational implementation for handwriting synthesis using Recurrent Neural Networks (RNNs). 

The handwriting synthesis in TypeScribe is based on the work presented in the paper **[Generating Sequences with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)** by Alex Graves. [Graves, A. (2013). *Generating sequences with recurrent neural networks*. arXiv preprint arXiv:1308.0850.]

---

Thank you for trying TypeScribe! ✍️
