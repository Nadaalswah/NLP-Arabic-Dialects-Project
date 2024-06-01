const logo = document.querySelectorAll("#logo path");
var loader = document.querySelector(".loader");
const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');
//console.log(logo);
for(let i = 0; i < logo.length; i++){
    console.log(`Letter ${i} is ${logo[i].getTotalLength()}`);
};
let list = document.querySelectorAll(".list");
function activelink() {
  list.forEach((item) => item.classList.remove("active"));
  this.classList.add("active");
}
list.forEach((item) => item.addEventListener("click", activelink));

// login and sign up
signUpButton.addEventListener('click', () => {
	container.classList.add("right-panel-active");
});
signInButton.addEventListener('click', () => {
	container.classList.remove("right-panel-active");
});
var li_items = document.querySelectorAll(".side_bar_bottom ul li");

li_items.forEach(function(li_main){
	li_main.addEventListener("click", function(){
		li_items.forEach(function(li){
			li.classList.remove("active");
		})
		li_main.classList.add("active");
	});
});
// stop Animatio 
/*window.addEventListener("load", fill);
function fill (){
    loader.className += "hidden";
}; */