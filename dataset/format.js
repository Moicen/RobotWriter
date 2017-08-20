const fs = require("fs");



const format = (origin) => {
	
	let lines = origin.split("\n");
	let content = [], title = true;
	var i = 0; line = "";
	for(i = 0; i < lines.length; i++){
		line = lines[i];
		
		if(title) {
			line = "\n" + line;
			console.log(title);
			title = false;
		} else {
			line = "    " + line;
		}
		if(/^\s+$/.test(line)) {
			title = true;
			console.log("line " + i + ": empty");
		} else {
			content.push(line);
		}
	}
	return content.join("\n");
	
}



fs.readFile("zheng.txt", "utf8", (err, data) => {
	if(err){
		console.log("error: ", err);
		return;
	}

	fs.writeFile("zheng-1.txt", format(data), (err) => {
		if(err){
			console.log("error: ", err);
			return;
		}
		console.log("format completed.");
	});

});