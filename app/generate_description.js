const fs = require('fs');
const path = require('path');

const outputFile = 'app_description.txt';
const rootDir = '.'; // Adjust this path as necessary

const excludedDirs = [
    // path.resolve('./python-interface/cpu_and_mobile'),
];

const excludedFiles = [
    path.resolve('./Readme.md'),
    path.resolve('./generate_description.js'),
];

function describeDirectory(dirPath, fileStream) {
    const files = fs.readdirSync(dirPath);

    files.forEach(file => {
        const fullPath = path.join(dirPath, file);
        const resolvedPath = path.resolve(fullPath);
        const stats = fs.statSync(fullPath);

        if (stats.isDirectory()) {
            if (!excludedDirs.includes(resolvedPath)) {
                fileStream.write(`\n=== Directory: ${fullPath} ===\n`);
                describeDirectory(fullPath, fileStream);
            } else {
                console.log(`Skipping directory: ${resolvedPath}`);
            }
        } else {
            if (!excludedFiles.includes(resolvedPath)) {
                const content = fs.readFileSync(fullPath, 'utf-8');
                fileStream.write(`\n--- File: ${fullPath} ---\n`);
                fileStream.write('Content:\n');
                fileStream.write('```js\n'); // Assuming most files are JavaScript; change if necessary
                fileStream.write(content);
                fileStream.write('\n```\n');
            } else {
                console.log(`Skipping file: ${resolvedPath}`);
            }
        }
    });
}

function generateDescription() {
    const fileStream = fs.createWriteStream(outputFile, { flags: 'w' });

    try {
        describeDirectory(rootDir, fileStream);
    } catch (error) {
        console.error('Error while generating description:', error);
    } finally {
        fileStream.end();
    }
}

generateDescription();