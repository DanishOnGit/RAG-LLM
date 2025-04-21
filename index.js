#!/usr/bin/env node
import { config } from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import chalk from "chalk";
import readline from "readline";
import https from "https";
import fetch from "node-fetch";

// Load environment variables
config();

// Create readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Promisify readline question
function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, (answer) => {
      resolve(answer);
    });
  });
}

async function generateEmbeddings(text) {
  try {
    // Create an https agent that ignores SSL certificate errors
    const httpsAgent = new https.Agent({
      rejectUnauthorized: false
    });

    const response = await fetch('https://api.jina.ai/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.JINA_API_KEY}`
      },
      body: JSON.stringify({
        model: 'jina-clip-v2',
        input: [{ text: text }]
      }),
      agent: httpsAgent
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const result = await response.json();
    console.log('Embedding generated successfully',result.data[0].embedding);
    return result.data[0].embedding;
  } catch (error) {
    console.error('Error generating embeddings:', error);
    // Fallback to a simple vector if the API call fails
    return new Array(512).fill(0);
  }
}

// ES Module equivalent for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.VITE_OPENAI_BASE_URL,
  defaultQuery: {
    "api-version": "2025-01-01-preview",
  },
  defaultHeaders: {
    "api-key": process.env.VITE_OPENAI_API_KEY,
  },
});

function loadDocuments() {
  const docsDir = path.join(__dirname, "knowledge-base");
  const files = fs
    .readdirSync(docsDir)
    .filter((file) => file.endsWith(".json"));
  return files.map((filename) => {
    try {
      const filePath = path.join(docsDir, filename);
      const content = JSON.parse(fs.readFileSync(filePath, "utf-8"));
      return { filename, content: content.data || "" };
    } catch (error) {
      console.error(`Error loading ${filename}:`, error);
      return { filename, content: "" };
    }
  });
}

async function retrieveRelevantDocsWithEmbeddings(query, documents) {
  try {
    console.log("Generating query embedding...");
    const queryEmbedding = await getEmbedding(query);
    
    if (!queryEmbedding || !Array.isArray(queryEmbedding)) {
      console.error("Failed to generate valid embedding for query");
      return [];
    }

    console.log("Comparing with document embeddings...");
    const docsWithScores = await Promise.all(
      documents.map(async (doc) => {
        try {
          const embedding = await getEmbedding(doc.content);
          const score = cosineSimilarity(queryEmbedding, embedding);
          return { ...doc, score };
        } catch (error) {
          console.error(`Error processing document ${doc.filename}:`, error);
          return { ...doc, score: 0 };
        }
      })
    );

    // Filter by relevance threshold and sort by score
    return docsWithScores
      .filter((doc) => doc.score > 0.5)
      .sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error("Error in retrieveRelevantDocsWithEmbeddings:", error);
    return [];
  }
}

async function getEmbedding(text) {
    console.log({text})
    return await generateEmbeddings(text);
//   const embedding = await openai.embeddings.create({
//     model: "text-embedding-3-small",
//     input: text,
//   });
//   return embedding.data[0].embedding;
}

function cosineSimilarity(vecA, vecB) {
  // Ensure vectors exist and have the same length
  if (!vecA || !vecB || !Array.isArray(vecA) || !Array.isArray(vecB)) {
    console.error('Invalid vectors provided to cosineSimilarity');
    return 0;
  }
  
  // Use the shorter length to avoid out-of-bounds errors
  const length = Math.min(vecA.length, vecB.length);
  
  if (length === 0) {
    return 0;
  }
  
  // Calculate dot product for the common length
  const dotProduct = Array.from({length}, (_, i) => vecA[i] * vecB[i])
    .reduce((sum, val) => sum + val, 0);
    
  // Calculate norms
  const normA = Math.sqrt(
    vecA.slice(0, length).reduce((sum, a) => sum + a * a, 0)
  );
  
  const normB = Math.sqrt(
    vecB.slice(0, length).reduce((sum, b) => sum + b * b, 0)
  );
  
  // Avoid division by zero
  if (normA === 0 || normB === 0) {
    return 0;
  }
  
  return dotProduct / (normA * normB);
}

async function generateAnswerWithOpenAI(query, context) {
  try {
    console.log("Generating answer using OpenAI...");
    const completion = await openai.chat.completions.create({
      model: "Proton", // or whatever model is available with your API key
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, just say 'I don't have enough information to answer this question.'"
        },
        {
          role: "user",
          content: `Context: ${context}\n\nQuestion: ${query}\n\nAnswer:`
        }
      ],
      temperature: 0.7,
      max_tokens: 500
    });

    return completion.choices[0].message.content;
  } catch (error) {
    console.error("Error generating answer with OpenAI:", error);
    return "Sorry, I encountered an error while generating an answer.";
  }
}

async function main() {
  const documents = loadDocuments();
  console.log(documents);
  const query = await question("Ask a question: ");

  const relevantDocs = await retrieveRelevantDocsWithEmbeddings(
    query,
    documents
  );
  if (relevantDocs.length === 0) {
    console.log("No relevant documents found.");
    rl.close();
    return;
  }

  const context = relevantDocs.map((doc) => doc.content).join("\n\n");
  const answer = await generateAnswerWithOpenAI(query, context);
  console.log(
    "\n📄 Context from:",
    relevantDocs.map((d) => d.filename).join(", ")
  );
  console.log("🧠 Answer:", answer);
  rl.close();
}
main();
