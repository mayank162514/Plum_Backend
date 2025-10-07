// const express = require('express');
// const multer = require('multer');
// const bodyParser = require('body-parser');
// const cors = require('cors');
// const { default: router } = require('./routes/amountRouter.js');

import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import multer from 'multer';
import router from './routes/amountRouter.js';

let GoogleGenerativeAI = null;
try {
  ({ GoogleGenerativeAI } = require('@google/generative-ai'));
} catch (e) {
  GoogleGenerativeAI = null;
}

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '5mb' }));
app.use(bodyParser.urlencoded({ extended: true }));

// multer setup for file uploads with size limit

const PORT = process.env.PORT || 3000;
app.use('/',router);
app.listen(PORT, () => {
  console.log(`AI Amount Detection backend running on port ${PORT}`);
});
