// The functions which run windows load event

window.addEventListener("load", () => {
  // Function to check whether script is running or not
  console.log("helloooooooooo");

  // Setting up the 2D canavas
  // saving canvas attributes in canvas variable
  // const canvas = document.querySelector("#canvas");
  // const ctx = canvas.getContext("2d");

  // // setting height and width of the canvas
  // canvas.height = 500; //window.innerHeight;
  // canvas.width = 500; //window.innerWidth;

  // // Making and setting "painting" Variable as false

  // let painting = false;

  // // Function to start drawing
  // function startPos(event) {
  //   painting = true;
  //   draw(event);
  // }

  // // Function to stop drawing
  // function stopPos() {
  //   painting = false;
  //   ctx.beginPath();
  // }

  // // Function to draw
  // function draw(event) {
  //   // Checking the condition of painting being true
  //   if (!painting) return;

  //   // Capturing the mouse coordinates in the variables
  //   var x = event.clientX;
  //   var y = event.clientY;

  //   // Setting the attributes of the canvas in respect to "line"
  //   ctx.lineWidth = 60; //89 was the previous value
  //   ctx.lineCap = "round";
  //   ctx.strokeStyle = "white";

  //   // Drawing line to the mouse coordinates
  //   ctx.lineTo(x, y);
  //   ctx.stroke();

  //   ctx.beginPath();
  //   ctx.moveTo(x, y);

  //   // Making html elements ---- "loader", "loader_text", "mousepos" visible while drawing on canvas
  //   document.getElementById("loader").style.visibility = "visible";
  //   document.getElementById("loader_text").style.visibility = "visible";
  //   document.getElementById("mousepos").style.visibility = "visible";
  // }

  // // Setting the initiation of "startPos", "stopPos", and "draw" function upon the different mouse activities
  // canvas.addEventListener("mousedown", startPos);
  // canvas.addEventListener("mouseup", stopPos);
  // canvas.addEventListener("mousemove", draw);

  // calling the "inactivityTime" function
  inactivityTime();

  // console.log(document.getElementById("pic_final").getAttribute("src"));

  if (document.getElementById("pic_final").getAttribute("src") != ""){
    document.getElementById("pic_final").style.visibility = "visible";
  } else {
    document.getElementById("pic_final").style.visibility = "hidden";
  }
});

// Function to show mouse coordinates on screen
// function showCoords(event) {
//   var x = event.clientX;
//   var y = event.clientY;
//   var coor = "Calculating Mouse Position - X coords: " + x + ", Y coords: " + y;
//   document.getElementById("mousepos").innerHTML = coor;
// }

// // Functtion to hide mouse coordinates from screen
// function clearCoor() {
//   document.getElementById("mousepos").innerHTML = "";
// }

// Function to download the drawn image on canvas
// function downloaded() {
//   var download = document.getElementById("downloader");
//   var image = document
//     .getElementById("canvas")
//     .toDataURL("image/jpeg")
//     .replace("image/jpeg", "image/octet-stream");
//   download.setAttribute("href", image);
// }

// Function to transfer the DataURL of canvas to the value of "link" element
// It transforms the image of the canvas in encoded strings
// and then assigns the encoded value of canvas to "link" value
// It further get transferred to the "app.py" file upon the submission of form
// on the click of the "predict" button
// function link_send() {
//   var link_transfer = document.getElementById("link");
//   var canvas = document.getElementById("canvas");
//   var dataURL = canvas.toDataURL();
//   link_transfer.setAttribute("value", dataURL);
// }

// Manipulating the position of loader(dog animation) upon the show of prediction text
// if (document.getElementById("answer").innerText != "") {
//   document.getElementById("loader").style.top = "25%";
// } else {
//   document.getElementById("loader").style.top = "37%";
// }

// Function to make the predict button blink upon the inactivity on screen
var inactivityTime = function () {
  // Variable to store inactivity time
  var time;

  // Events to restart/reset the inactivity time
  document.onload = resetTimer;
  document.onmousemove = resetTimer;
  document.onmousedown = resetTimer; // touchscreen presses
  document.ontouchstart = resetTimer;
  document.onclick = resetTimer; // touchpad clicks
  document.onkeypress = resetTimer;
  document.addEventListener("scroll", resetTimer, true); // improved; see comments

  // Function to add the glow animation to the predict button
  function btn_blink() {
    document.getElementById("submit_link").style.animation =
      "glowing 1300ms infinite";
    //location.href = 'logout.html'
  }

  // Function to reset time variable upon any activity on the screen
  function resetTimer() {
    // Method to clear out the value stored in the time variable upon the initiation of the function
    clearTimeout(time);
    // Assigning the Method "setTimeout" to the variable time
    // Setting the limit of 6 seconds to start the "btn_blink" function which sets the animation on the predict button
    time = setTimeout(btn_blink, 6000);
    // 1000 milliseconds = 1 second
  }
};


var loadFile = function (event) {
  var output = document.getElementById('output');
  output.src = URL.createObjectURL(event.target.files[0]);
  document.getElementById("output").style.visibility = "visible";
  output.onload = function () {
    URL.revokeObjectURL(output.src) // free memory
  }
};

function picture_show(){
  document.getElementById("pic_final").style.visibility = "visible";
}