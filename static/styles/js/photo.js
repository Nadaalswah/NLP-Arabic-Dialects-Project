var li_items = document.querySelectorAll(".side_bar_bottom ul li");

li_items.forEach(function(li_main){
  li_main.addEventListener("click", function(){
    li_items.forEach(function(li){
      li.classList.remove("active");
    })
    li_main.classList.add("active");
  });
});

/*let canves = document.querySelector("#canvas"); 
let context = canves.getContext("2d");
let video = document.querySelector("#video");

if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
    navigator.mediaDevices.getUserMedia({video : true}).then((stream) => {
        video.srcObject = stream;
        video.play();
    });
}

document.getElementById("snap").addEventListener("click",() => {
    context.drawImage(video, 0, 0, 640, 480);
}); */


handleDownloadImages = (event) => {
    if (event.target.files && event.target.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function (e) {
                        const img = document.querySelector("#uploaded")
                        console.log(img)
                        img.setAttribute("src", e.target.result);
                        img.classList.remove('hidden');
                        document.querySelector("#file-upload").classList.add('hidden')
                        document.querySelector(".btn-block").classList.remove('hidden')
        };
        reader.readAsDataURL(event.target.files[0]);
    }
}

document.querySelector("#upload-btn").onclick = (event) => {
 document.querySelector('.progress-bar span').classList.add('activate')
 window.setTimeout(() => { 
     event.target.innerHTML = 'done'

 } , 2500)
}