document.getElementById('generate').addEventListener('click', async () => {
  const numItems = parseInt(document.getElementById('count').value);
  const option = document.getElementById('option').value;
  const output = document.getElementById('output');
  output.innerHTML = '';

  try {
    const response = await fetch('lorem.txt');
    if (!response.ok) {
      throw new Error('Failed to load lorem text');
    }
    const loremText = await response.text();

    if (option === 'paragraphs') {
      // Generate paragraphs
        const paraArray = loremText.split('\n\n');
        const selectedParas = paraArray.slice(0, numItems).join('\n\n');
        const paragraph = document.createElement('p');
        paragraph.textContent = selectedParas;
        output.appendChild(paragraph);
    } else if (option === 'words') {
        // Generate words
        const wordsArray = loremText.split(' ');
        const selectedWords = wordsArray.slice(0, numItems).join(' ');
        const paragraph = document.createElement('p');
        paragraph.textContent = selectedWords;
        output.appendChild(paragraph);
    }
  } catch (error) {
    console.error('Error:', error);
    output.textContent = 'Failed to load the lorem ipsum text.';
  }
});
