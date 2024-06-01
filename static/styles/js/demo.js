let list = document.querySelectorAll(".list");
function activelink() {
  list.forEach((item) => item.classList.remove("active"));
  this.classList.add("active");
}
list.forEach((item) => item.addEventListener("click", activelink));
// Demo 
let video = document.querySelector("#videoElement");
if (navigator.mediaDevices.getUserMedia){
    navigator.mediaDevices.getUserMedia({video: true})
    .then(function (stream) {
    video.srcObject = stream;
    })
    .catch (function (error) {
        console.log("Something went wrong!");
    })

}else {
    console.log("getUserMedia not supported!");
}
var li_items = document.querySelectorAll(".side_bar_bottom ul li");

li_items.forEach(function(li_main){
	li_main.addEventListener("click", function(){
		li_items.forEach(function(li){
			li.classList.remove("active");
		})
		li_main.classList.add("active");
	});
});
/*******
 * //

const constraints = {
    video: true
  };
  
  const video = document.querySelector('video');
  
  function handleSuccess(stream) {
    video.srcObject = stream;
  }
  
  function handleError(error) {
    console.error('Reeeejected!', error);
  }
  
  navigator.mediaDevices.getUserMedia(constraints).
    then(handleSuccess).catch(handleError);
  
  //
  // Selecting a an audio and video source: 
  //
  
  const videoElement = document.querySelector('video');
  const audioSelect = document.querySelector('select#audioSource');
  const videoSelect = document.querySelector('select#videoSource');
  
  navigator.mediaDevices.enumerateDevices()
    .then(gotDevices).then(getStream).catch(handleError);
  
  audioSelect.onchange = getStream;
  videoSelect.onchange = getStream;
  
  function gotDevices(deviceInfos) {
    for (var i = 0; i !== deviceInfos.length; ++i) {
      var deviceInfo = deviceInfos[i];
      var option = document.createElement('option');
      option.value = deviceInfo.deviceId;
      if (deviceInfo.kind === 'audioinput') {
        option.text = deviceInfo.label ||
          'microphone ' + (audioSelect.length + 1);
        audioSelect.appendChild(option);
      } else if (deviceInfo.kind === 'videoinput') {
        option.text = deviceInfo.label || 'camera ' +
          (videoSelect.length + 1);
        videoSelect.appendChild(option);
      } else {
        console.log('Found one other kind of source/device: ', deviceInfo);
      }
    }
  }
  
  function getStream() {
    if (window.stream) {
    window.stream.getTracks().forEach(function(track) {
        track.stop();
    });
}

    var constraints = {
    audio: {
        deviceId: {exact: audioSelect.value}
    },
    video: {
        deviceId: {exact: videoSelect.value}
    }
    };
    navigator.mediaDevices.getUserMedia(constraints).
        then(gotStream).catch(handleError);
} 
function gotStream(stream) {
    window.stream = stream; // make stream available to console
    videoElement.srcObject = stream;
}
function handleError(error) {
    console.error('Error: ', error);
}
 */
// Feature detection - navigator.mediaDevices.getUserMedia 
// 

/*function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}  
if (hasGetUserMedia()) {
    alert('Your browser supports access to your webcam. Click Okay or Yes and in the next pop-up Allow this webpage use your camera. If you have more than one camera you will be asked which should be used.');
} else {
    alert('This page\'s video capture functionality is not supported by your browser. Please update your browser or use a differnt browser like Chrome, FireFox or Opera.');
}
  //
  // Gaining access to an input device 
  //
  /* instead of {video: true} indicating basic access to video is required, more requirements (or constraints) on the returned media stream can be specified e.g. requiring the stream to be HD. */
