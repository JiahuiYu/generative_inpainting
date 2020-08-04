var pos = {
    drawable: false,
    x: -1,
    y: -1
}

var canvas, ctx; 

function drawing(){
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext('2d');

    var img1 = new Image;
    img1.src = URL.createObjectURL(document.getElementById("image").files[0]);
    img1.onload = function() {
        URL.revokeObjectURL(img1.src) // free memory
        ctx.drawImage(img1,0,0)
    }
    
    canvas.addEventListener("mousedown", listener);
    canvas.addEventListener("mouseup",listener);
    canvas.addEventListener("mousemove", listener);
    canvas.addEventListener("mouseup", listener);
}

function listener(event){
    switch(event.type){
        case "mousedown":
            initDraw(event);
            break;
        case "mousemove":
            if(pos.drawable)
                draw(event);
            break;
        case "mouseout":
        case "mouseup":
            finishDraw();
            break;
    }
}

function initDraw(event) {
    ctx.beginPath();
    pos.drawable = true;
    var coors = getPosition(event);
    pos.X = coors.X;
    pos.Y = coors.Y;
    ctx.moveTo(pos.X, pos.Y);
}

function draw(event) {
    var cooors = getPostion(event);
    ctx.lineTo(coors.X, coors.Y);
    pos.X = coors.X;
    pos.Y = coors.Y;
    ctx.lineWidth = 5;
    ctx.stroke();
}

function finishDraw() {
    pos.drawable = false;
    pos.X = -1;
    pos.Y = -1;
}

function getPosition(event) {
    var x = event.pageX - canvas.offsetLeft;
    var y = event.pageY - canvas.offsetTop;
    return {
        X:x,
        Y:y,
    }
}