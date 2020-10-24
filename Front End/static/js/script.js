// The functions which run windows load event

window.addEventListener("load", () => {
  // Function to check whether script is running or not
  // console.log("helloooooooooo");

  // calling the "inactivityTime" function
  inactivityTime();

  // console.log(document.getElementById("pic_final").getAttribute("src"));

  // Checking whether the final image source is empty in order to hide the boundary of the image
  if (document.getElementById("pic_final").getAttribute("src") != ""){
    document.getElementById("pic_final").style.display = "reset";
  } else {
    document.getElementById("pic_final").style.display = "none";
  }
});

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

// To display the chosed file before uploading it
var loadFile = function (event) {
  var output = document.getElementById('output');
  output.src = URL.createObjectURL(event.target.files[0]);
  document.getElementById("output").style.display = "revert";

  document.getElementById("direction").style.display = "none";

  output.onload = function () {
    URL.revokeObjectURL(output.src) // free memory
  }
};