import { QALanguageModel } from "../index";
import fs from "fs/promises";

async function main() {
    try {
        // Read the CSV data
        const text = await fs.readFile(
            "./models/v0-2/test/datasets-2.csv",
            "utf-8"
        );

        // Parse the CSV data
        const rows = text.trim().split("\n");
        const headers = rows[0].split(",");

        // Get the indices of the req and res columns
        const reqIndex = headers.indexOf("req");
        const resIndex = headers.indexOf("res");

        if (reqIndex === -1 || resIndex === -1) {
            throw new Error('CSV must contain "req" and "res" columns');
        }

        // Extract question-answer pairs
        const qaPairs = [];
        for (let i = 1; i < rows.length; i++) {
            // Handle possible commas within quoted fields
            const currentRow = [];
            let inQuotes = false;
            let currentField = "";

            for (const char of rows[i]) {
                if (char === '"') {
                    inQuotes = !inQuotes;
                } else if (char === "," && !inQuotes) {
                    currentRow.push(currentField);
                    currentField = "";
                } else {
                    currentField += char;
                }
            }
            currentRow.push(currentField); // Add the last field

            // If we have both question and answer, add to our pairs
            if (currentRow.length > Math.max(reqIndex, resIndex)) {
                qaPairs.push({
                    req: currentRow[reqIndex],
                    res: currentRow[resIndex],
                });
            }
        }

        console.log(`Parsed ${qaPairs.length} QA pairs`);

        // Create and train the model
        const model = new QALanguageModel(2);
        model.batchTrainQA(
            qaPairs.map((x) => ({ question: x.req, answers: x.res }))
        );
        console.log(model.getModelStats());

        console.log("Training completed successfully");

        // Optional: Test the model with a question
        const testQuestion = "머해?";
        const response = await model
            .beginConversation()
            .addUserMessage(testQuestion);
        console.log(`Test Q: ${testQuestion}`);
        console.log(`Model A: ${response}`);
    } catch (error) {
        console.error("Error during training:", error);
    }
}

main();
