var pos = {
    drawable: false,
    x: -1,
    y: -1
}

var canvas, ctx; 

var pointList = []
var points = []

function drawing(img){
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext('2d');
    ctx.clearRect(0,0, canvas.width, canvas.height);
    
    canvas.width = img.width;
    canvas.height = img.height;
    
    ctx.drawImage(img, 0, 0, img.width, img.height);

    ctx.strokeStyle = "white";
    ctx.lineWidth = 20;
    
    var pointerSize = document.getElementById("pointerSize");
    pointerSize.value = 20;

    points = [];
    pointList = [];

    canvas.addEventListener("mousedown", listener);
    canvas.addEventListener("mouseup",listener);
    canvas.addEventListener("mousemove", listener);
    canvas.addEventListener("mouseup", listener);

    img.style.display = "none";    
}

function setLineWidthSize(size) {
    ctx.lineWidth = size;
}

function clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
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

    points.push({
        x: pos.X,
        y: pos.Y,
        mode:"begin"
    })
}

function draw(event) {
    var coors = getPosition(event);
    ctx.lineTo(coors.X, coors.Y);
    pos.X = coors.X;
    pos.Y = coors.Y;
    ctx.stroke();

    points.push({
        x: pos.X,
        y: pos.Y,
        mode: "draw"
    })
}

function finishDraw() {
    pos.drawable = false;
    var coors = getPosition(event);
    pos.X = coors.X;
    pos.Y = coors.Y;

    points.push({
        x: pos.X,
        y: pos.Y,
        mode: "end"
    })

    pointList.push(points);
    points = [];
}

function getPosition(event) {
    var x = event.pageX - canvas.offsetLeft;
    var y = event.pageY - canvas.offsetTop;
    return {
        X:x,
        Y:y,
    }
}
function redrawAll(){

    ctx.clearRect(0,0,canvas.width,canvas.height);

    var img = document.getElementById('hiddenImage');
    img.style.display = "block";

    ctx.drawImage(img, 0, 0, img.width, img.height);
    img.style.display = "none";

    for(var i=0;i<pointList.length;i++){

        var points=pointList[i];
        
        for(var j=0; j<points.length; j++){
            var pt = points[j];
            if(pt.mode=="begin"){
                ctx.beginPath();
                ctx.moveTo(pt.x,pt.y);
            }
            ctx.lineTo(pt.x,pt.y);
            if(pt.mode=="end" || (i==points.length-1)){
                ctx.stroke();
            }
        }
        
    }
    
}

function undo() {
    pointList.pop();
    redrawAll();
}

function createMask() {
    ctx.clearRect(0,0, canvas.width, canvas.height);

    for(var i=0;i<pointList.length;i++){

        var points=pointList[i];
        
        for(var j=0; j<points.length; j++){
            var pt = points[j];
            if(pt.mode=="begin"){
                ctx.beginPath();
                ctx.moveTo(pt.x,pt.y);
            }
            ctx.lineTo(pt.x,pt.y);
            if(pt.mode=="end" || (i==points.length-1)){
                ctx.stroke();
            }
        }
        
    }

    return saveCanvas();
}

function saveCanvas() {
    var img = canvas.toDataURL();
    return img;
}

function showCanvas() {
    canvas.style.display = 'block';
    
    var undo = document.getElementById("undo");
    undo.style.display = 'inline-block';

    var pointerSize = document.getElementById("pointerSize");
    pointerSize.style.display = 'inline-block';
    var brushWidth = document.getElementById("Brushwidth");
    brushWidth.style.display = 'inline-block';
}

function hideCanvas() {
    canvas.style.display = "none";

    var undo = document.getElementById("undo");
    undo.style.display = 'none';   
    var pointerSize = document.getElementById("pointerSize");
    pointerSize.style.display = 'none';
    var brushWidth = document.getElementById("Brushwidth");
    brushWidth.style.display = 'none';
}