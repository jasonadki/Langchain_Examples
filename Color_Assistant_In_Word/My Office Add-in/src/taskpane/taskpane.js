Office.onReady((info) => {
  if (info.host === Office.HostType.Word) {
    document.getElementById("askBot").onclick = () => tryCatch(getAnswerFromBot);
    document.getElementById("sideload-msg").style.display = "none";
    document.getElementById("app-body").style.display = "flex";
    document.addEventListener('DOMContentLoaded', function() {
      var spinnerElement = document.querySelector('.ms-Spinner');
      if (spinnerElement) {
          new fabric['Spinner'](spinnerElement);
      }
    });
  }
});

async function getAnswerFromBot() {
  
  await Word.run(async (context) => {
    // Get the current selection.
    var selection = context.document.getSelection();
    
    // Load the text property.
    selection.load('text');

    // Synchronize the document state by executing the queued commands.
    await context.sync();

    // Fetch the highlighted text.
    let highlightedText = selection.text;

    // Send the highlighted text to the bot.
    const response = await fetch("http://localhost:5000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: highlightedText,
        pdf: "results.json" // Assuming you're sending a static PDF, modify this accordingly.
      }),
    });

    const responseData = await response.json();
    // hideSpinner();

    // Parse inner JSON
    const parsedResponse = JSON.parse(responseData.response);

    console.log(responseData);

    // Update the Word document with the color.
    selection.insertText(`(${parsedResponse.color}) ${highlightedText}`, Word.InsertLocation.replace);
    await context.sync();

    // Display the bot's reasoning on the task pane.
    document.getElementById("result").innerText = `Reasoning: ${parsedResponse.reasoning}`;

    // Render the documents
    let docsOutput = '';
    for (let doc of responseData.documents) {
        docsOutput += `<strong>${doc.description} = (${doc.code})</strong> - ${doc.document_name}<br>`;
    }
    document.getElementById("references").innerHTML = docsOutput;

  });
}

/** Default helper for invoking an action and handling errors. */
function tryCatch(callback) {
  try {
    callback();
  } catch (error) {
    console.error(error);
  }
}

function showSpinner() {
  document.getElementById('spinner').style.display = 'block';
}

function hideSpinner() {
  document.getElementById('spinner').style.display = 'none';
}
