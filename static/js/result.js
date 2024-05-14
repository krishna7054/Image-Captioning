window.onload = function () {
    const imageHandle = document.getElementById('uploadedImage')
    const paraHandle = document.getElementById('filename-para')
    imageHandle.src = '../static/uploads/' + paraHandle.innerText
}

var back=document.getElementById('back');
back.addEventListener("click",function(){
  location.href="/";
});
