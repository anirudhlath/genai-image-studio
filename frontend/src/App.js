import React, { useState } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Button, 
  TextField, 
  Grid, 
  Paper, 
  CircularProgress 
} from '@mui/material';
import { Upload } from '@mui/icons-material';
import Dropzone from 'react-dropzone';
import axios from 'axios';

function App() {
  const [uploadedImages, setUploadedImages] = useState([]);
  const [sessionID, setSessionID] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleUpload = async (acceptedFiles) => {
    setIsLoading(true);
    const formData = new FormData();
    acceptedFiles.forEach((file, index) => {
      formData.append('files', file);
    });

    try {
      const response = await axios.post('http://localhost:8000/upload-images/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setSessionID(response.data.session_id);
      setUploadedImages(acceptedFiles);
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!sessionID) return;
    
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/train-model/', {
        session_id: sessionID
      });
      setTrainingStatus(response.data.message);
    } catch (error) {
      console.error('Error training model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!prompt) return;
    
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/generate/', {
        prompt: prompt
      });
      setGeneratedImage(response.data);
    } catch (error) {
      console.error('Error generating image:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        DreamBooth Image Generator
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Upload Images
            </Typography>
            <Dropzone onDrop={handleUpload}>
              {({ getRootProps, getInputProps }) => (
                <Box
                  {...getRootProps()}
                  sx={{
                    border: '2px dashed grey',
                    p: 3,
                    textAlign: 'center',
                    cursor: 'pointer',
                  }}
                >
                  <input {...getInputProps()} />
                  <Upload fontSize="large" />
                  <Typography variant="body1">
                    Drag & drop some files here, or click to select files
                  </Typography>
                </Box>
              )}
            </Dropzone>
          </Paper>
        </Grid>

        {/* Training Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Training
            </Typography>
            <Button
              variant="contained"
              onClick={handleTrain}
              disabled={!sessionID || isLoading}
              sx={{ mb: 2 }}
            >
              Train Model
            </Button>
            {trainingStatus && (
              <Typography variant="body1" color="success.main">
                {trainingStatus}
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Generation Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Generate Image
            </Typography>
            <TextField
              fullWidth
              label="Enter Prompt"
              variant="outlined"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              onClick={handleGenerate}
              disabled={!prompt || isLoading}
              sx={{ mb: 2 }}
            >
              Generate
            </Button>
            {generatedImage && (
              <img
                src={`http://localhost:8000${generatedImage}`}
                alt="Generated"
                style={{ maxWidth: '100%', maxHeight: '400px' }}
              />
            )}
          </Paper>
        </Grid>
      </Grid>
      {isLoading && (
        <Box
          sx={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        >
          <CircularProgress />
        </Box>
      )}
    </Container>
  );
}

export default App;
