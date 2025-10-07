import express from 'express';
import multer from 'multer';
import { classify, extract, finalize, fullPipeline, normalizeValues } from '../controllers/amountController.js';

const upload = multer({ dest: 'uploads/', limits: { fileSize: 5 * 1024 * 1024 } });
const router = express.Router();

router.post('/step1', upload.single('file'),extract);

// Step2 route: normalize raw tokens
router.post('/step2', normalizeValues);

// Step3 route: classify
router.post('/step3', classify);

// Step4 route: finalize
router.post('/step4', finalize);

// Full pipeline (image + text optional)
router.post('/process', upload.single('file'), fullPipeline);

// Health
router.get('/health', (req, res) => res.json({ status: 'ok' }));

export default router;